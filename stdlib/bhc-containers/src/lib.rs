//! BHC Containers Library - Rust FFI support for LLVM codegen
//!
//! Provides C-callable functions for container operations used by
//! generated LLVM code. Containers are opaque heap-allocated objects.
//!
//! At the LLVM level, all values are pointer-sized (`*mut u8`).
//! Map/Set use i64-cast pointer comparison for key ordering.

#![warn(missing_docs)]
#![allow(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ptr;
use std::sync::Mutex;

// ========================================================================
// Opaque container types
// ========================================================================

/// Opaque Map type: BTreeMap<i64, *mut u8> behind a Box.
type RtsMap = BTreeMap<i64, *mut u8>;

/// Opaque Set type: BTreeSet<i64> behind a Box.
type RtsSet = BTreeSet<i64>;

// ========================================================================
// Key canonicalization
// ========================================================================

/// Interned canonical keys: content -> the first pointer seen for it.
static INTERNED_KEYS: Mutex<Option<HashMap<Vec<u8>, i64>>> = Mutex::new(None);

/// A `BhcText`'s header is `[data_ptr][offset][byte_len]`, 24 bytes.
/// Canonical definition: `bhc-text`'s `text.rs`.
const TEXT_HEADER_SIZE: usize = 24;

/// Read the active bytes of a `BhcText`.
unsafe fn text_key_bytes(text: *const u8) -> Vec<u8> {
    let data = unsafe { *(text as *const *const u8) };
    let off = unsafe { *((text as *const u64).add(1)) } as usize;
    let len = unsafe { *((text as *const u64).add(2)) } as usize;
    if data.is_null() || len > (1 << 30) {
        return Vec::new();
    }
    let _ = TEXT_HEADER_SIZE;
    unsafe { std::slice::from_raw_parts(data.add(off), len) }.to_vec()
}

/// Read the bytes of a `[Char]` cons list. Layout: nil = `[0]`,
/// cons = `[1][head][tail]`.
unsafe fn char_list_key_bytes(mut p: *const u8) -> Vec<u8> {
    let mut out = Vec::new();
    let mut guard = 0usize;
    while !p.is_null() {
        let tag = unsafe { *(p as *const i64) };
        if tag != 1 {
            break;
        }
        let head = unsafe { *(p.add(8) as *const u64) } as u32;
        if let Some(ch) = char::from_u32(head) {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
        p = unsafe { *(p.add(16) as *const *const u8) };
        guard += 1;
        if guard > (1 << 24) {
            break;
        }
    }
    out
}

/// Canonicalize a container key so that equal CONTENTS become the same key.
///
/// A Map or Set orders its keys as machine words, which is right for an `Int`
/// and wrong for anything boxed: two equal `Text`s are two different handles,
/// so `M.lookup k (M.fromList [(k, v)])` answered `Nothing` for every `Map
/// Text v` in pandoc. Interning gives every equal content the first pointer
/// ever seen for it, which restores lookup, membership and deletion.
///
/// `kind` says how to read the key, and codegen only passes a non-zero one
/// where it knows the key's type — an unknown key keeps today's behaviour
/// rather than risking a dereference of a large integer.
///
/// Ordering among interned keys is by that first pointer, not by content, so
/// `toList` on a boxed-key map is not in key order.
///
/// # Safety
/// For `kind` 1 the key must be a live `BhcText`; for `kind` 2 a `[Char]`
/// cons list. Kind 0 does not dereference.
#[no_mangle]
pub unsafe extern "C" fn bhc_container_key(kind: i64, raw: i64) -> i64 {
    if kind == 0 || raw == 0 {
        return raw;
    }
    let ptr = raw as *const u8;
    let bytes = match kind {
        1 => unsafe { text_key_bytes(ptr) },
        2 => unsafe { char_list_key_bytes(ptr) },
        _ => return raw,
    };
    let mut guard = match INTERNED_KEYS.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let table = guard.get_or_insert_with(HashMap::new);
    *table.entry(bytes).or_insert(raw)
}

/// Container objects already re-keyed, so the walk happens once each.
static CANONICALIZED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether `ptr` still needs re-keying, marking it done.
fn claim_for_canon(ptr: *const u8) -> bool {
    let mut guard = match CANONICALIZED.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.get_or_insert_with(HashSet::new).insert(ptr as usize)
}

/// Re-key a map so its boxed keys compare by content.
///
/// `fromList` cannot always see what kind of key it is building with — the
/// list's element type is often erased by the time Core reaches codegen —
/// while the operation that later looks a key up usually can, from the map's
/// own type. So the map is re-keyed on first use by a caller that knows,
/// in place: replacing each key with its canonical form preserves the map's
/// contents, and doing it to a shared map is what makes every alias agree.
///
/// # Safety
/// `map_ptr` must be null or a live `RtsMap` whose keys match `kind`.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_canon(map_ptr: *mut u8, kind: i64) {
    if map_ptr.is_null() || kind == 0 || !claim_for_canon(map_ptr) {
        return;
    }
    let m = unsafe { &mut *(map_ptr as *mut RtsMap) };
    let entries: Vec<(i64, *mut u8)> = m.iter().map(|(k, v)| (*k, *v)).collect();
    let mut rekeyed = BTreeMap::new();
    for (k, v) in entries {
        rekeyed.insert(unsafe { bhc_container_key(kind, k) }, v);
    }
    *m = rekeyed;
}

/// Re-key a set so its boxed elements compare by content. See
/// [`bhc_map_canon`].
///
/// # Safety
/// `set_ptr` must be null or a live `RtsSet` whose elements match `kind`.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_canon(set_ptr: *mut u8, kind: i64) {
    if set_ptr.is_null() || kind == 0 || !claim_for_canon(set_ptr) {
        return;
    }
    let s = unsafe { &mut *(set_ptr as *mut RtsSet) };
    let elems: Vec<i64> = s.iter().copied().collect();
    let mut rekeyed = BTreeSet::new();
    for e in elems {
        rekeyed.insert(unsafe { bhc_container_key(kind, e) });
    }
    *s = rekeyed;
}

// ========================================================================
// Data.Map operations
// ========================================================================

/// Create an empty map.
#[no_mangle]
pub extern "C" fn bhc_map_empty() -> *mut u8 {
    let m: Box<RtsMap> = Box::default();
    Box::into_raw(m) as *mut u8
}

/// Create a singleton map.
#[no_mangle]
pub extern "C" fn bhc_map_singleton(key: i64, value: *mut u8) -> *mut u8 {
    let mut m = BTreeMap::new();
    m.insert(key, value);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Check if map is empty. Returns 1 if null, 0 otherwise.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_null(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() {
        return 1;
    }
    let m = &*(map_ptr as *const RtsMap);
    if m.is_empty() {
        1
    } else {
        0
    }
}

/// Get the size of a map.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_size(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    let m = &*(map_ptr as *const RtsMap);
    m.len() as i64
}

/// Check if a key is a member of the map. Returns 1 if member, 0 otherwise.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_member(key: i64, map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    let m = &*(map_ptr as *const RtsMap);
    if m.contains_key(&key) {
        1
    } else {
        0
    }
}

/// Lookup a key in the map. Returns the value pointer or null if not found.
/// The caller must wrap in Just/Nothing.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call. The
/// returned value pointer is borrowed from the map and is only valid while the
/// map remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_lookup(key: i64, map_ptr: *mut u8) -> *mut u8 {
    if map_ptr.is_null() {
        return ptr::null_mut();
    }
    let m = &*(map_ptr as *const RtsMap);
    match m.get(&key) {
        Some(&v) => v,
        None => ptr::null_mut(),
    }
}

/// Find with default: return the value for key, or default if not found.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call. The
/// `default` pointer is returned unchanged when the key is absent.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_find_with_default(
    default: *mut u8,
    key: i64,
    map_ptr: *mut u8,
) -> *mut u8 {
    if map_ptr.is_null() {
        return default;
    }
    let m = &*(map_ptr as *const RtsMap);
    match m.get(&key) {
        Some(&v) => v,
        None => default,
    }
}

/// Insert a key-value pair into the map. Returns a new map (COW).
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input map is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_insert(key: i64, value: *mut u8, map_ptr: *mut u8) -> *mut u8 {
    let mut m = if map_ptr.is_null() {
        BTreeMap::new()
    } else {
        (*(map_ptr as *const RtsMap)).clone()
    };
    m.insert(key, value);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Delete a key from the map. Returns a new map (COW).
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input map is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_delete(key: i64, map_ptr: *mut u8) -> *mut u8 {
    if map_ptr.is_null() {
        return bhc_map_empty();
    }
    let mut m = (*(map_ptr as *const RtsMap)).clone();
    m.remove(&key);
    Box::into_raw(Box::new(m)) as *mut u8
}

/// Union of two maps (left-biased). Returns a new map.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_map_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_union(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    let mut result = if map1.is_null() {
        BTreeMap::new()
    } else {
        (*(map1 as *const RtsMap)).clone()
    };
    if !map2.is_null() {
        let m2 = &*(map2 as *const RtsMap);
        for (&k, &v) in m2.iter() {
            result.entry(k).or_insert(v);
        }
    }
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Intersection of two maps (left-biased). Returns a new map.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_map_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_intersection(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    if map1.is_null() || map2.is_null() {
        return bhc_map_empty();
    }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    let result: RtsMap = m1
        .iter()
        .filter(|(k, _)| m2.contains_key(k))
        .map(|(&k, &v)| (k, v))
        .collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Difference of two maps. Returns a new map with keys in m1 but not m2.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_map_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_difference(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    if map1.is_null() {
        return bhc_map_empty();
    }
    if map2.is_null() {
        return Box::into_raw(Box::new((*(map1 as *const RtsMap)).clone())) as *mut u8;
    }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    let result: RtsMap = m1
        .iter()
        .filter(|(k, _)| !m2.contains_key(k))
        .map(|(&k, &v)| (k, v))
        .collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Get the keys of a map as a count + array.
/// Returns the number of keys. Writes key array to `out_keys` if non-null.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_keys_count(map_ptr: *mut u8) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    let m = &*(map_ptr as *const RtsMap);
    m.len() as i64
}

/// Get a key at index from the map (for iteration).
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
/// Out-of-range `index` values yield 0 rather than reading out of bounds.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_key_at(map_ptr: *mut u8, index: i64) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    let m = &*(map_ptr as *const RtsMap);
    m.keys().nth(index as usize).copied().unwrap_or(0)
}

/// Get a value at index from the map (for iteration).
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_map_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
/// Out-of-range `index` values yield null. The returned value pointer is
/// borrowed from the map and is only valid while the map remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_value_at(map_ptr: *mut u8, index: i64) -> *mut u8 {
    if map_ptr.is_null() {
        return ptr::null_mut();
    }
    let m = &*(map_ptr as *const RtsMap);
    m.values()
        .nth(index as usize)
        .copied()
        .unwrap_or(ptr::null_mut())
}

/// Check if map1 is a submap of map2.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_map_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn bhc_map_is_submap_of(map1: *mut u8, map2: *mut u8) -> i64 {
    if map1.is_null() {
        return 1;
    }
    if map2.is_null() {
        return 0;
    }
    let m1 = &*(map1 as *const RtsMap);
    let m2 = &*(map2 as *const RtsMap);
    if m1.keys().all(|k| m2.contains_key(k)) {
        1
    } else {
        0
    }
}

// ========================================================================
// Data.Set operations
// ========================================================================

/// Create an empty set.
#[no_mangle]
pub extern "C" fn bhc_set_empty() -> *mut u8 {
    Box::into_raw(Box::new(BTreeSet::<i64>::new())) as *mut u8
}

/// Create a singleton set.
#[no_mangle]
pub extern "C" fn bhc_set_singleton(value: i64) -> *mut u8 {
    let mut s = BTreeSet::new();
    s.insert(value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Check if set is empty.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_null(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 1;
    }
    let s = &*(set_ptr as *const RtsSet);
    if s.is_empty() {
        1
    } else {
        0
    }
}

/// Get the size of a set.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_size(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    s.len() as i64
}

/// Check if a value is a member of the set.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_member(value: i64, set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    if s.contains(&value) {
        1
    } else {
        0
    }
}

/// Insert a value into the set. Returns a new set (COW).
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input set is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_insert(value: i64, set_ptr: *mut u8) -> *mut u8 {
    let mut s = if set_ptr.is_null() {
        BTreeSet::new()
    } else {
        (*(set_ptr as *const RtsSet)).clone()
    };
    s.insert(value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Delete a value from the set. Returns a new set (COW).
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input set is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete(value: i64, set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() {
        return bhc_set_empty();
    }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    s.remove(&value);
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Union of two sets. Returns a new set.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_set_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_union(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    let s1 = if set1.is_null() {
        BTreeSet::new()
    } else {
        (*(set1 as *const RtsSet)).clone()
    };
    let empty = BTreeSet::new();
    let s2 = if set2.is_null() {
        &empty
    } else {
        &*(set2 as *const RtsSet)
    };
    let result: BTreeSet<i64> = s1.union(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Intersection of two sets. Returns a new set.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_set_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_intersection(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    if set1.is_null() || set2.is_null() {
        return bhc_set_empty();
    }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    let result: BTreeSet<i64> = s1.intersection(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Difference of two sets. Returns a new set.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_set_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_difference(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    if set1.is_null() {
        return bhc_set_empty();
    }
    if set2.is_null() {
        return Box::into_raw(Box::new((*(set1 as *const RtsSet)).clone())) as *mut u8;
    }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    let result: BTreeSet<i64> = s1.difference(s2).copied().collect();
    Box::into_raw(Box::new(result)) as *mut u8
}

/// Check if set1 is a subset of set2.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_set_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_is_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    if set1.is_null() {
        return 1;
    }
    if set2.is_null() {
        return 0;
    }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    if s1.is_subset(s2) {
        1
    } else {
        0
    }
}

/// Check if set1 is a proper subset of set2.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_set_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_is_proper_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    if set1.is_null() {
        return if set2.is_null() { 0 } else { 1 };
    }
    if set2.is_null() {
        return 0;
    }
    let s1 = &*(set1 as *const RtsSet);
    let s2 = &*(set2 as *const RtsSet);
    if s1.is_subset(s2) && s1.len() < s2.len() {
        1
    } else {
        0
    }
}

/// Get count of elements in set (for iteration).
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_elem_count(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    s.len() as i64
}

/// Get element at index from the set (for iteration).
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
/// Out-of-range `index` values yield 0 rather than reading out of bounds.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_elem_at(set_ptr: *mut u8, index: i64) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().nth(index as usize).copied().unwrap_or(0)
}

/// Find the minimum element of a set. Returns 0 if empty.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_find_min(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().next().copied().unwrap_or(0)
}

/// Find the maximum element of a set. Returns 0 if empty.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_find_max(set_ptr: *mut u8) -> i64 {
    if set_ptr.is_null() {
        return 0;
    }
    let s = &*(set_ptr as *const RtsSet);
    s.iter().next_back().copied().unwrap_or(0)
}

/// Delete the minimum element. Returns a new set.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input set is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete_min(set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() {
        return bhc_set_empty();
    }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    if let Some(&min) = s.iter().next() {
        s.remove(&min);
    }
    Box::into_raw(Box::new(s)) as *mut u8
}

/// Delete the maximum element. Returns a new set.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_set_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input set is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_set_delete_max(set_ptr: *mut u8) -> *mut u8 {
    if set_ptr.is_null() {
        return bhc_set_empty();
    }
    let mut s = (*(set_ptr as *const RtsSet)).clone();
    if let Some(&max) = s.iter().next_back() {
        s.remove(&max);
    }
    Box::into_raw(Box::new(s)) as *mut u8
}

// ========================================================================
// Data.IntMap operations (identical to Map since Map also uses i64 keys)
// ========================================================================

/// Create an empty IntMap.
#[no_mangle]
pub extern "C" fn bhc_intmap_empty() -> *mut u8 {
    bhc_map_empty()
}

/// Create a singleton IntMap.
#[no_mangle]
pub extern "C" fn bhc_intmap_singleton(key: i64, value: *mut u8) -> *mut u8 {
    bhc_map_singleton(key, value)
}

/// Check if IntMap is empty.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_null(map_ptr: *mut u8) -> i64 {
    bhc_map_null(map_ptr)
}

/// Get IntMap size.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_size(map_ptr: *mut u8) -> i64 {
    bhc_map_size(map_ptr)
}

/// Check IntMap membership.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_member(key: i64, map_ptr: *mut u8) -> i64 {
    bhc_map_member(key, map_ptr)
}

/// IntMap lookup.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call. The returned value pointer is
/// borrowed from the map and is only valid while the map remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_lookup(key: i64, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_lookup(key, map_ptr)
}

/// IntMap findWithDefault.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call. The `default` pointer is
/// returned unchanged when the key is absent.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_find_with_default(
    default: *mut u8,
    key: i64,
    map_ptr: *mut u8,
) -> *mut u8 {
    bhc_map_find_with_default(default, key, map_ptr)
}

/// IntMap insert.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call. The input map is cloned, not
/// mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_insert(key: i64, value: *mut u8, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_insert(key, value, map_ptr)
}

/// IntMap delete.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call. The input map is cloned, not
/// mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_delete(key: i64, map_ptr: *mut u8) -> *mut u8 {
    bhc_map_delete(key, map_ptr)
}

/// IntMap union.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_union(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_union(map1, map2)
}

/// IntMap intersection.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_intersection(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_intersection(map1, map2)
}

/// IntMap difference.
///
/// # Safety
///
/// `map1` and `map2` must each be either null or point to a live `RtsMap`
/// previously returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_difference(map1: *mut u8, map2: *mut u8) -> *mut u8 {
    bhc_map_difference(map1, map2)
}

/// IntMap keys count.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_keys_count(map_ptr: *mut u8) -> i64 {
    bhc_map_keys_count(map_ptr)
}

/// IntMap key at index.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_key_at(map_ptr: *mut u8, index: i64) -> i64 {
    bhc_map_key_at(map_ptr, index)
}

/// IntMap value at index.
///
/// # Safety
///
/// `map_ptr` must either be null or point to a live `RtsMap` previously
/// returned by one of the `bhc_intmap_*`/`bhc_map_*` constructors and must
/// remain valid for the duration of the call. The returned value pointer is
/// borrowed from the map and is only valid while the map remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_intmap_value_at(map_ptr: *mut u8, index: i64) -> *mut u8 {
    bhc_map_value_at(map_ptr, index)
}

// ========================================================================
// Data.IntSet operations (identical to Set)
// ========================================================================

/// Create an empty IntSet.
#[no_mangle]
pub extern "C" fn bhc_intset_empty() -> *mut u8 {
    bhc_set_empty()
}

/// Create a singleton IntSet.
#[no_mangle]
pub extern "C" fn bhc_intset_singleton(value: i64) -> *mut u8 {
    bhc_set_singleton(value)
}

/// Check if IntSet is empty.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_null(set_ptr: *mut u8) -> i64 {
    bhc_set_null(set_ptr)
}

/// Get IntSet size.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_size(set_ptr: *mut u8) -> i64 {
    bhc_set_size(set_ptr)
}

/// Check IntSet membership.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_member(value: i64, set_ptr: *mut u8) -> i64 {
    bhc_set_member(value, set_ptr)
}

/// IntSet insert.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call. The input set is cloned, not
/// mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_insert(value: i64, set_ptr: *mut u8) -> *mut u8 {
    bhc_set_insert(value, set_ptr)
}

/// IntSet delete.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call. The input set is cloned, not
/// mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_delete(value: i64, set_ptr: *mut u8) -> *mut u8 {
    bhc_set_delete(value, set_ptr)
}

/// IntSet union.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_intset_*`/`bhc_set_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_union(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_union(set1, set2)
}

/// IntSet intersection.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_intset_*`/`bhc_set_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_intersection(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_intersection(set1, set2)
}

/// IntSet difference.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_intset_*`/`bhc_set_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_difference(set1: *mut u8, set2: *mut u8) -> *mut u8 {
    bhc_set_difference(set1, set2)
}

/// IntSet isSubsetOf.
///
/// # Safety
///
/// `set1` and `set2` must each be either null or point to a live `RtsSet`
/// previously returned by one of the `bhc_intset_*`/`bhc_set_*` constructors
/// and must remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_is_subset_of(set1: *mut u8, set2: *mut u8) -> i64 {
    bhc_set_is_subset_of(set1, set2)
}

/// IntSet element count.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_elem_count(set_ptr: *mut u8) -> i64 {
    bhc_set_elem_count(set_ptr)
}

/// IntSet element at index.
///
/// # Safety
///
/// `set_ptr` must either be null or point to a live `RtsSet` previously
/// returned by one of the `bhc_intset_*`/`bhc_set_*` constructors and must
/// remain valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_intset_elem_at(set_ptr: *mut u8, index: i64) -> i64 {
    bhc_set_elem_at(set_ptr, index)
}

// ========================================================================
// Data.Sequence operations (Vec-backed)
// ========================================================================

/// Opaque Seq type: Vec<*mut u8> behind a Box.
type RtsSeq = Vec<*mut u8>;

/// Create an empty sequence.
#[no_mangle]
pub extern "C" fn bhc_seq_empty() -> *mut u8 {
    Box::into_raw(Box::new(Vec::<*mut u8>::new())) as *mut u8
}

/// Create a singleton sequence.
#[no_mangle]
pub extern "C" fn bhc_seq_singleton(elem: *mut u8) -> *mut u8 {
    Box::into_raw(Box::new(vec![elem])) as *mut u8
}

/// Check if sequence is empty. Returns 1 if empty, 0 otherwise.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_null(seq_ptr: *mut u8) -> i64 {
    if seq_ptr.is_null() {
        return 1;
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        1
    } else {
        0
    }
}

/// Get the length of a sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_length(seq_ptr: *mut u8) -> i64 {
    if seq_ptr.is_null() {
        return 0;
    }
    let s = &*(seq_ptr as *const RtsSeq);
    s.len() as i64
}

/// Index into a sequence. Panics on out-of-bounds.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call.
/// Out-of-range `idx` values yield null rather than reading out of bounds. The
/// returned element pointer is borrowed from the sequence and is only valid
/// while the sequence remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_index(seq_ptr: *mut u8, idx: i64) -> *mut u8 {
    if seq_ptr.is_null() {
        return ptr::null_mut();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let i = idx as usize;
    if i < s.len() {
        s[i]
    } else {
        ptr::null_mut()
    }
}

/// Lookup by index, returning null if out-of-bounds (for Maybe wrapping).
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call. The
/// returned element pointer is borrowed from the sequence and is only valid
/// while the sequence remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_lookup(idx: i64, seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return ptr::null_mut();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let i = idx as usize;
    if i < s.len() {
        s[i]
    } else {
        ptr::null_mut()
    }
}

/// Prepend an element (`<|`). Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_cons(elem: *mut u8, seq_ptr: *mut u8) -> *mut u8 {
    let mut v = if seq_ptr.is_null() {
        Vec::new()
    } else {
        (*(seq_ptr as *const RtsSeq)).clone()
    };
    v.insert(0, elem);
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Append an element (`|>`). Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_snoc(seq_ptr: *mut u8, elem: *mut u8) -> *mut u8 {
    let mut v = if seq_ptr.is_null() {
        Vec::new()
    } else {
        (*(seq_ptr as *const RtsSeq)).clone()
    };
    v.push(elem);
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Concatenate two sequences (`><`). Returns a new sequence.
///
/// # Safety
///
/// `seq1` and `seq2` must each be either null or point to a live `RtsSeq`
/// previously returned by one of the `bhc_seq_*` constructors. Both pointees
/// must remain valid and not be mutated concurrently for the duration of the
/// call; they are read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_append(seq1: *mut u8, seq2: *mut u8) -> *mut u8 {
    let mut v1 = if seq1.is_null() {
        Vec::new()
    } else {
        (*(seq1 as *const RtsSeq)).clone()
    };
    if !seq2.is_null() {
        let s2 = &*(seq2 as *const RtsSeq);
        v1.extend_from_slice(s2);
    }
    Box::into_raw(Box::new(v1)) as *mut u8
}

/// Take first n elements. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_take(n: i64, seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let take_n = (n as usize).min(s.len());
    Box::into_raw(Box::new(s[..take_n].to_vec())) as *mut u8
}

/// Drop first n elements. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_drop(n: i64, seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let drop_n = (n as usize).min(s.len());
    Box::into_raw(Box::new(s[drop_n..].to_vec())) as *mut u8
}

/// Reverse a sequence. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_reverse(seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let mut v = s.clone();
    v.reverse();
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Update element at index. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
/// Out-of-range `idx` values leave the cloned sequence unchanged.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_update(idx: i64, elem: *mut u8, seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let mut v = s.clone();
    let i = idx as usize;
    if i < v.len() {
        v[i] = elem;
    }
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Insert element at index. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact. The
/// insertion index is clamped to the sequence length.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_insert_at(idx: i64, elem: *mut u8, seq_ptr: *mut u8) -> *mut u8 {
    let mut v = if seq_ptr.is_null() {
        Vec::new()
    } else {
        (*(seq_ptr as *const RtsSeq)).clone()
    };
    let i = (idx as usize).min(v.len());
    v.insert(i, elem);
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Delete element at index. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// cloned rather than mutated, so the input sequence is left intact.
/// Out-of-range `idx` values leave the cloned sequence unchanged.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_delete_at(idx: i64, seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    let mut v = s.clone();
    let i = idx as usize;
    if i < v.len() {
        v.remove(i);
    }
    Box::into_raw(Box::new(v)) as *mut u8
}

/// Get element count (for toList iteration). Same as length.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_elem_count(seq_ptr: *mut u8) -> i64 {
    bhc_seq_length(seq_ptr)
}

/// Get element at index (for toList iteration). Same as index.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call. The returned element pointer is borrowed from the
/// sequence and is only valid while the sequence remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_elem_at(seq_ptr: *mut u8, idx: i64) -> *mut u8 {
    bhc_seq_index(seq_ptr, idx)
}

/// Replicate: create a sequence of n copies of an element.
#[no_mangle]
pub extern "C" fn bhc_seq_replicate(n: i64, elem: *mut u8) -> *mut u8 {
    let count = if n < 0 { 0 } else { n as usize };
    Box::into_raw(Box::new(vec![elem; count])) as *mut u8
}

/// ViewL tag: 0 if empty, 1 if non-empty.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewl_tag(seq_ptr: *mut u8) -> i64 {
    if seq_ptr.is_null() {
        return 0;
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        0
    } else {
        1
    }
}

/// ViewL head: first element (undefined if empty).
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call. Returns null for null/empty inputs. The returned
/// element pointer is borrowed from the sequence and is only valid while the
/// sequence remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewl_head(seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return ptr::null_mut();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        ptr::null_mut()
    } else {
        s[0]
    }
}

/// ViewL tail: all elements after first. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewl_tail(seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        return bhc_seq_empty();
    }
    Box::into_raw(Box::new(s[1..].to_vec())) as *mut u8
}

/// ViewR tag: 0 if empty, 1 if non-empty.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewr_tag(seq_ptr: *mut u8) -> i64 {
    if seq_ptr.is_null() {
        return 0;
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        0
    } else {
        1
    }
}

/// ViewR last: last element (undefined if empty).
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors and must remain valid for
/// the duration of the call. Returns null for null/empty inputs. The returned
/// element pointer is borrowed from the sequence and is only valid while the
/// sequence remains live.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewr_last(seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return ptr::null_mut();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        ptr::null_mut()
    } else {
        *s.last().unwrap()
    }
}

/// ViewR init: all elements except last. Returns a new sequence.
///
/// # Safety
///
/// `seq_ptr` must either be null or point to a live `RtsSeq` previously
/// returned by one of the `bhc_seq_*` constructors. The pointee must remain
/// valid and not be mutated concurrently for the duration of the call; it is
/// read/cloned, not mutated.
#[no_mangle]
pub unsafe extern "C" fn bhc_seq_viewr_init(seq_ptr: *mut u8) -> *mut u8 {
    if seq_ptr.is_null() {
        return bhc_seq_empty();
    }
    let s = &*(seq_ptr as *const RtsSeq);
    if s.is_empty() {
        return bhc_seq_empty();
    }
    Box::into_raw(Box::new(s[..s.len() - 1].to_vec())) as *mut u8
}
