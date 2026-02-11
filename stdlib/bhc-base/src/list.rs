//! List operations
//!
//! C-callable RTS functions for list manipulation used by
//! LLVM-generated code. Lists are represented as heap-allocated
//! ADT nodes with Nil (tag=0) and Cons (tag=1, head, tail) layout.

use std::alloc::{alloc, Layout};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// ADT helpers (internal)
// ---------------------------------------------------------------------------

/// Read the tag (i64) at offset 0 of an ADT node.
unsafe fn get_tag(ptr: *mut u8) -> i64 {
    *(ptr as *const i64)
}

/// Read field at `index` (0-based) starting at offset 8.
unsafe fn get_field(ptr: *mut u8, index: usize) -> *mut u8 {
    *(ptr.add(8 + index * 8) as *const *mut u8)
}

/// Allocate a Nil node (tag=0, 8 bytes).
unsafe fn alloc_nil() -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(8, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 0;
    ptr
}

/// Allocate a Cons node (tag=1, head at offset 8, tail at offset 16).
unsafe fn alloc_cons(head: *mut u8, tail: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(24, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 1;
    *(ptr.add(8) as *mut *mut u8) = head;
    *(ptr.add(16) as *mut *mut u8) = tail;
    ptr
}

// ---------------------------------------------------------------------------
// Bool / Maybe / Tuple helpers (internal)
// ---------------------------------------------------------------------------

/// Extract bool from either tagged-int-as-pointer (0/1 from ==, <, >)
/// or Bool ADT (heap pointer with tag 0=False, 1=True from even, odd, etc.).
/// Heap pointers are always >> 1 so we can distinguish the two representations.
unsafe fn extract_bool(val: *mut u8) -> bool {
    let v = val as usize;
    if v <= 1 {
        v != 0
    } else {
        get_tag(val) != 0
    }
}

/// Call a 2-arg equality closure and extract the Bool result.
/// BHC closures are flat: fn(env, arg1, arg2) -> result.
unsafe fn call_eq_closure(eq_fn: *mut u8, a: *mut u8, b: *mut u8) -> bool {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(eq_fn as *const *mut u8));
    extract_bool(fn_ptr(eq_fn, a, b))
}

/// Allocate Nothing ADT (tag=0, 8 bytes).
unsafe fn alloc_nothing() -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(8, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 0;
    ptr
}

/// Allocate Just x ADT (tag=1, field at offset 8, 16 bytes).
unsafe fn alloc_just(value: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(16, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 1;
    *(ptr.add(8) as *mut *mut u8) = value;
    ptr
}

/// Allocate a 2-tuple (tag=0, fst at offset 8, snd at offset 16, 24 bytes).
unsafe fn alloc_tuple(fst: *mut u8, snd: *mut u8) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(24, 8);
    let ptr = alloc(layout);
    *(ptr as *mut i64) = 0;
    *(ptr.add(8) as *mut *mut u8) = fst;
    *(ptr.add(16) as *mut *mut u8) = snd;
    ptr
}

// ---------------------------------------------------------------------------
// Conversion helpers (internal)
// ---------------------------------------------------------------------------

/// Collect a linked list into a `Vec<*mut u8>`.
unsafe fn list_to_vec(mut list: *mut u8) -> Vec<*mut u8> {
    let mut vec = Vec::new();
    loop {
        if get_tag(list) == 0 {
            break;
        }
        vec.push(get_field(list, 0));
        list = get_field(list, 1);
    }
    vec
}

/// Build a linked list from a slice, preserving order.
unsafe fn vec_to_list(slice: &[*mut u8]) -> *mut u8 {
    let mut result = alloc_nil();
    for &elem in slice.iter().rev() {
        result = alloc_cons(elem, result);
    }
    result
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Sort a linked list by comparing elements as `i64` (pointer cast).
///
/// Converts to `Vec`, sorts, and rebuilds a new list.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_sort(list: *mut u8) -> *mut u8 {
    let mut vec = list_to_vec(list);
    vec.sort_by_key(|&e| e as i64);
    vec_to_list(&vec)
}

/// Sort a linked list using a comparison closure.
///
/// `cmp_fn` is a pointer to a closure struct whose first field
/// (offset 0) is the code pointer with signature
/// `extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8`.
/// The return value encodes Ordering as LT=-1, EQ=0, GT=1 cast to
/// `*mut u8`.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_sort_by(cmp_fn: *mut u8, list: *mut u8) -> *mut u8 {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(cmp_fn as *const *mut u8));

    let mut vec = list_to_vec(list);
    vec.sort_by(|&a, &b| {
        let result = fn_ptr(cmp_fn, a, b) as i64;
        match result {
            r if r < 0 => std::cmp::Ordering::Less,
            0 => std::cmp::Ordering::Equal,
            _ => std::cmp::Ordering::Greater,
        }
    });
    vec_to_list(&vec)
}

/// Remove duplicate elements from a list (by `i64` value comparison).
///
/// Preserves the order of first occurrences.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_nub(list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    let mut seen = HashSet::new();
    let deduped: Vec<*mut u8> = vec
        .into_iter()
        .filter(|&e| seen.insert(e as i64))
        .collect();
    vec_to_list(&deduped)
}

/// Group consecutive equal elements into sublists.
///
/// `group [1,1,2,2,2,3] = [[1,1],[2,2,2],[3]]`
#[no_mangle]
pub unsafe extern "C" fn bhc_list_group(list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    if vec.is_empty() {
        return alloc_nil();
    }

    let mut groups: Vec<Vec<*mut u8>> = Vec::new();
    let mut current_group: Vec<*mut u8> = vec![vec[0]];

    for &elem in &vec[1..] {
        if elem as i64 == *current_group.last().unwrap() as i64 {
            current_group.push(elem);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(elem);
        }
    }
    groups.push(current_group);

    // Build list of lists: convert each group to a sublist, then wrap
    let sublists: Vec<*mut u8> = groups.iter().map(|g| vec_to_list(g)).collect();
    vec_to_list(&sublists)
}

/// Concatenate a list of lists with a separator list between each.
///
/// `intercalate ", " ["a","b","c"] = "a, b, c"` (for general lists).
#[no_mangle]
pub unsafe extern "C" fn bhc_list_intercalate(sep: *mut u8, lists: *mut u8) -> *mut u8 {
    let sep_vec = list_to_vec(sep);
    let outer = list_to_vec(lists);

    if outer.is_empty() {
        return alloc_nil();
    }

    let mut result: Vec<*mut u8> = Vec::new();
    for (i, &sublist_ptr) in outer.iter().enumerate() {
        if i > 0 {
            result.extend_from_slice(&sep_vec);
        }
        let sublist = list_to_vec(sublist_ptr);
        result.extend_from_slice(&sublist);
    }
    vec_to_list(&result)
}

/// Transpose a list of lists.
///
/// `transpose [[1,2],[3,4],[5,6]] = [[1,3,5],[2,4,6]]`
///
/// Follows Haskell semantics: rows of different lengths are handled
/// by skipping missing elements (shorter rows are ignored once
/// exhausted).
#[no_mangle]
pub unsafe extern "C" fn bhc_list_transpose(lists: *mut u8) -> *mut u8 {
    let rows: Vec<Vec<*mut u8>> = list_to_vec(lists)
        .into_iter()
        .map(|row_ptr| list_to_vec(row_ptr))
        .collect();

    if rows.is_empty() {
        return alloc_nil();
    }

    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut columns: Vec<Vec<*mut u8>> = Vec::with_capacity(max_cols);

    for col in 0..max_cols {
        let column: Vec<*mut u8> = rows
            .iter()
            .filter_map(|row| row.get(col).copied())
            .collect();
        columns.push(column);
    }

    let sublists: Vec<*mut u8> = columns.iter().map(|c| vec_to_list(c)).collect();
    vec_to_list(&sublists)
}

/// Sort a list using a key-extraction closure (decorate-sort-undecorate).
///
/// `sortOn f xs` sorts `xs` by comparing `f x` values as `i64`.
/// `key_fn` is a 1-arg closure: `fn(env, elem) -> key`.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_sort_on(key_fn: *mut u8, list: *mut u8) -> *mut u8 {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(key_fn as *const *mut u8));

    let vec = list_to_vec(list);
    let mut decorated: Vec<(i64, *mut u8)> = vec
        .iter()
        .map(|&elem| {
            let key = fn_ptr(key_fn, elem);
            (key as i64, elem)
        })
        .collect();
    decorated.sort_by_key(|&(k, _)| k);
    let sorted: Vec<*mut u8> = decorated.into_iter().map(|(_, e)| e).collect();
    vec_to_list(&sorted)
}

/// Remove duplicates using a custom equality closure.
///
/// `nubBy eq xs` keeps the first occurrence of each element,
/// removing later elements for which `eq earlier later` is True.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_nub_by(eq_fn: *mut u8, list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    let mut result: Vec<*mut u8> = Vec::new();
    for &elem in &vec {
        let already = result.iter().any(|&kept| call_eq_closure(eq_fn, kept, elem));
        if !already {
            result.push(elem);
        }
    }
    vec_to_list(&result)
}

/// Group consecutive elements using a custom equality closure.
///
/// `groupBy eq xs` groups consecutive elements where `eq a b` is True.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_group_by(eq_fn: *mut u8, list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    if vec.is_empty() {
        return alloc_nil();
    }

    let mut groups: Vec<Vec<*mut u8>> = Vec::new();
    let mut current_group: Vec<*mut u8> = vec![vec[0]];

    for &elem in &vec[1..] {
        if call_eq_closure(eq_fn, *current_group.last().unwrap(), elem) {
            current_group.push(elem);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(elem);
        }
    }
    groups.push(current_group);

    let sublists: Vec<*mut u8> = groups.iter().map(|g| vec_to_list(g)).collect();
    vec_to_list(&sublists)
}

/// Delete the first element matching by a custom equality closure.
///
/// `deleteBy eq x xs` removes the first `y` in `xs` where `eq x y` is True.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_delete_by(
    eq_fn: *mut u8,
    val: *mut u8,
    list: *mut u8,
) -> *mut u8 {
    let vec = list_to_vec(list);
    let mut result: Vec<*mut u8> = Vec::new();
    let mut found = false;
    for &elem in &vec {
        if !found && call_eq_closure(eq_fn, val, elem) {
            found = true;
        } else {
            result.push(elem);
        }
    }
    vec_to_list(&result)
}

/// Union of two lists using a custom equality closure.
///
/// `unionBy eq xs ys = xs ++ [y | y <- ys, not (any (eq y) xs')]`
/// where `xs'` grows as elements from `ys` are added.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_union_by(
    eq_fn: *mut u8,
    xs: *mut u8,
    ys: *mut u8,
) -> *mut u8 {
    let xs_vec = list_to_vec(xs);
    let ys_vec = list_to_vec(ys);
    let mut result = xs_vec.clone();
    for &y in &ys_vec {
        let already = result.iter().any(|&x| call_eq_closure(eq_fn, x, y));
        if !already {
            result.push(y);
        }
    }
    vec_to_list(&result)
}

/// Intersection of two lists using a custom equality closure.
///
/// `intersectBy eq xs ys = [x | x <- xs, any (eq x) ys]`
#[no_mangle]
pub unsafe extern "C" fn bhc_list_intersect_by(
    eq_fn: *mut u8,
    xs: *mut u8,
    ys: *mut u8,
) -> *mut u8 {
    let xs_vec = list_to_vec(xs);
    let ys_vec = list_to_vec(ys);
    let result: Vec<*mut u8> = xs_vec
        .into_iter()
        .filter(|&x| ys_vec.iter().any(|&y| call_eq_closure(eq_fn, x, y)))
        .collect();
    vec_to_list(&result)
}

/// Strip a prefix from a list, returning `Just remainder` or `Nothing`.
///
/// Elements are compared as `i64` (pointer cast).
#[no_mangle]
pub unsafe extern "C" fn bhc_list_strip_prefix(
    prefix: *mut u8,
    list: *mut u8,
) -> *mut u8 {
    let mut p = prefix;
    let mut l = list;
    loop {
        if get_tag(p) == 0 {
            // Prefix exhausted — return Just remaining
            return alloc_just(l);
        }
        if get_tag(l) == 0 {
            // List exhausted before prefix — return Nothing
            return alloc_nothing();
        }
        let p_head = get_field(p, 0);
        let l_head = get_field(l, 0);
        if p_head as i64 != l_head as i64 {
            return alloc_nothing();
        }
        p = get_field(p, 1);
        l = get_field(l, 1);
    }
}

/// Insert an element into a sorted list at the correct position.
///
/// Elements are compared as `i64` (pointer cast).
/// `insert 3 [1,2,4,5] = [1,2,3,4,5]`
#[no_mangle]
pub unsafe extern "C" fn bhc_list_insert(val: *mut u8, list: *mut u8) -> *mut u8 {
    let vec = list_to_vec(list);
    let v = val as i64;
    let pos = vec.iter().position(|&e| (e as i64) > v).unwrap_or(vec.len());
    let mut result = vec;
    result.insert(pos, val);
    vec_to_list(&result)
}

/// Left-to-right accumulating map.
///
/// `mapAccumL f acc xs` calls `f acc x` for each element left-to-right.
/// The closure returns a 2-tuple `(new_acc, y)`. Returns `(final_acc, ys)`.
/// `f` is a 2-arg closure: `fn(env, acc, x) -> tuple`.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_map_accum_l(
    f: *mut u8,
    acc: *mut u8,
    list: *mut u8,
) -> *mut u8 {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(f as *const *mut u8));

    let vec = list_to_vec(list);
    let mut current_acc = acc;
    let mut ys: Vec<*mut u8> = Vec::with_capacity(vec.len());

    for &x in &vec {
        let tuple = fn_ptr(f, current_acc, x);
        current_acc = get_field(tuple, 0);
        ys.push(get_field(tuple, 1));
    }

    let ys_list = vec_to_list(&ys);
    alloc_tuple(current_acc, ys_list)
}

/// Right-to-left accumulating map.
///
/// `mapAccumR f acc xs` calls `f acc x` for each element right-to-left.
/// The closure returns a 2-tuple `(new_acc, y)`. Returns `(final_acc, ys)`.
/// `f` is a 2-arg closure: `fn(env, acc, x) -> tuple`.
#[no_mangle]
pub unsafe extern "C" fn bhc_list_map_accum_r(
    f: *mut u8,
    acc: *mut u8,
    list: *mut u8,
) -> *mut u8 {
    let fn_ptr: extern "C" fn(*mut u8, *mut u8, *mut u8) -> *mut u8 =
        std::mem::transmute(*(f as *const *mut u8));

    let vec = list_to_vec(list);
    let mut current_acc = acc;
    let mut ys: Vec<*mut u8> = Vec::with_capacity(vec.len());

    for &x in vec.iter().rev() {
        let tuple = fn_ptr(f, current_acc, x);
        current_acc = get_field(tuple, 0);
        ys.push(get_field(tuple, 1));
    }

    ys.reverse();
    let ys_list = vec_to_list(&ys);
    alloc_tuple(current_acc, ys_list)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn make_list(elems: &[i64]) -> *mut u8 {
        let ptrs: Vec<*mut u8> = elems.iter().map(|&v| v as *mut u8).collect();
        vec_to_list(&ptrs)
    }

    unsafe fn collect_i64(list: *mut u8) -> Vec<i64> {
        list_to_vec(list).into_iter().map(|p| p as i64).collect()
    }

    unsafe fn collect_nested(list: *mut u8) -> Vec<Vec<i64>> {
        list_to_vec(list)
            .into_iter()
            .map(|sub| collect_i64(sub))
            .collect()
    }

    #[test]
    fn test_sort_empty() {
        unsafe {
            let list = alloc_nil();
            let sorted = bhc_list_sort(list);
            assert_eq!(collect_i64(sorted), Vec::<i64>::new());
        }
    }

    #[test]
    fn test_sort() {
        unsafe {
            let list = make_list(&[3, 1, 4, 1, 5, 9, 2, 6]);
            let sorted = bhc_list_sort(list);
            assert_eq!(collect_i64(sorted), vec![1, 1, 2, 3, 4, 5, 6, 9]);
        }
    }

    #[test]
    fn test_nub() {
        unsafe {
            let list = make_list(&[1, 2, 3, 2, 1, 4]);
            let deduped = bhc_list_nub(list);
            assert_eq!(collect_i64(deduped), vec![1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_nub_empty() {
        unsafe {
            let list = alloc_nil();
            let deduped = bhc_list_nub(list);
            assert_eq!(collect_i64(deduped), Vec::<i64>::new());
        }
    }

    #[test]
    fn test_group() {
        unsafe {
            let list = make_list(&[1, 1, 2, 2, 2, 3]);
            let groups = bhc_list_group(list);
            assert_eq!(
                collect_nested(groups),
                vec![vec![1, 1], vec![2, 2, 2], vec![3]]
            );
        }
    }

    #[test]
    fn test_group_empty() {
        unsafe {
            let list = alloc_nil();
            let groups = bhc_list_group(list);
            assert_eq!(collect_nested(groups), Vec::<Vec<i64>>::new());
        }
    }

    #[test]
    fn test_intercalate() {
        unsafe {
            // intercalate [0] [[1,2],[3,4],[5,6]] = [1,2,0,3,4,0,5,6]
            let sep = make_list(&[0]);
            let a = make_list(&[1, 2]);
            let b = make_list(&[3, 4]);
            let c = make_list(&[5, 6]);
            let lists = vec_to_list(&[a, b, c]);
            let result = bhc_list_intercalate(sep, lists);
            assert_eq!(collect_i64(result), vec![1, 2, 0, 3, 4, 0, 5, 6]);
        }
    }

    #[test]
    fn test_transpose() {
        unsafe {
            // transpose [[1,2],[3,4],[5,6]] = [[1,3,5],[2,4,6]]
            let a = make_list(&[1, 2]);
            let b = make_list(&[3, 4]);
            let c = make_list(&[5, 6]);
            let lists = vec_to_list(&[a, b, c]);
            let result = bhc_list_transpose(lists);
            assert_eq!(
                collect_nested(result),
                vec![vec![1, 3, 5], vec![2, 4, 6]]
            );
        }
    }

    #[test]
    fn test_transpose_ragged() {
        unsafe {
            // transpose [[1,2,3],[4]] = [[1,4],[2],[3]]
            let a = make_list(&[1, 2, 3]);
            let b = make_list(&[4]);
            let lists = vec_to_list(&[a, b]);
            let result = bhc_list_transpose(lists);
            assert_eq!(
                collect_nested(result),
                vec![vec![1, 4], vec![2], vec![3]]
            );
        }
    }
}
