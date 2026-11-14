//! Environment for variable bindings during evaluation.
//!
//! The environment maps variable IDs to their values. Extending it gives a
//! new environment and leaves the old one as it was, and a closure keeps
//! the environment it was made in, so the two must share structure.
//!
//! An environment is a hash map of older bindings, shared by reference,
//! and a short list of the bindings added since, newest first. Extending
//! adds to the list; when the list grows past [`FLATTEN_AT`] the two are
//! merged into a new map. A lookup walks the list and then asks the map.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::VarId;

use super::value::Value;

/// The number of recent bindings an environment keeps in its list before
/// merging them into its map. Merging copies the map, so it should be rare;
/// walking the list is every lookup, so the list should be short.
const FLATTEN_AT: usize = 32;

/// A binding added since the environment's map was built.
#[derive(Debug)]
struct Recent {
    id: VarId,
    value: Value,
    older: Option<Arc<Recent>>,
}

/// An environment mapping variables to values.
///
/// The environment is immutable and uses structural sharing for efficiency.
/// When a closure captures an environment, only reference counts are
/// incremented.
#[derive(Clone, Debug, Default)]
pub struct Env {
    /// The bindings added since `older` was built, newest first. A binding
    /// here shadows one of the same variable further down or in `older`.
    recent: Option<Arc<Recent>>,
    /// The length of the `recent` list.
    recent_len: usize,
    /// The bindings from before that.
    older: Arc<HashMap<VarId, Value>>,
}

impl Env {
    /// Creates an empty environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The recent bindings, newest first.
    fn recent(&self) -> impl Iterator<Item = &Recent> {
        std::iter::successors(self.recent.as_deref(), |binding| binding.older.as_deref())
    }

    /// Looks up a variable in the environment.
    #[must_use]
    pub fn lookup(&self, id: VarId) -> Option<&Value> {
        self.recent()
            .find(|binding| binding.id == id)
            .map(|binding| &binding.value)
            .or_else(|| self.older.get(&id))
    }

    /// Extends the environment with a new binding.
    ///
    /// Returns a new environment with the binding added.
    /// The original environment is unchanged.
    #[must_use]
    pub fn extend(&self, id: VarId, value: Value) -> Self {
        if self.recent_len >= FLATTEN_AT {
            return self.flattened_with([(id, value)]);
        }
        Self {
            recent: Some(Arc::new(Recent {
                id,
                value,
                older: self.recent.clone(),
            })),
            recent_len: self.recent_len + 1,
            older: Arc::clone(&self.older),
        }
    }

    /// Extends the environment with multiple bindings.
    #[must_use]
    pub fn extend_many(&self, bindings: impl IntoIterator<Item = (VarId, Value)>) -> Self {
        let bindings: Vec<(VarId, Value)> = bindings.into_iter().collect();
        if self.recent_len + bindings.len() > FLATTEN_AT {
            return self.flattened_with(bindings);
        }
        let mut env = self.clone();
        for (id, value) in bindings {
            env = env.extend(id, value);
        }
        env
    }

    /// This environment and `bindings` as one map, with an empty list.
    fn flattened_with(&self, bindings: impl IntoIterator<Item = (VarId, Value)>) -> Self {
        let mut map = HashMap::clone(&self.older);
        // Oldest first, so that a newer binding of a variable wins.
        let recent: Vec<&Recent> = self.recent().collect();
        for binding in recent.into_iter().rev() {
            map.insert(binding.id, binding.value.clone());
        }
        map.extend(bindings);
        Self {
            recent: None,
            recent_len: 0,
            older: Arc::new(map),
        }
    }

    /// Returns the number of bindings in the environment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys().count()
    }

    /// Returns true if the environment is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.recent.is_none() && self.older.is_empty()
    }

    /// Returns all `VarIds` in the environment (for debugging).
    pub fn keys(&self) -> impl Iterator<Item = &VarId> {
        let mut seen: HashSet<&VarId> = HashSet::new();
        let keys: Vec<&VarId> = self
            .recent()
            .map(|binding| &binding.id)
            .chain(self.older.keys())
            .filter(|id| seen.insert(*id))
            .collect();
        keys.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use bhc_index::Idx;

    use super::*;

    #[test]
    fn test_env_lookup() {
        let env = Env::new();
        let id = VarId::new(0);
        let val = Value::Int(42);

        let env2 = env.extend(id, val.clone());
        assert!(env.lookup(id).is_none());
        assert!(matches!(env2.lookup(id), Some(Value::Int(42))));
    }

    #[test]
    fn test_env_extend_many() {
        let env = Env::new();
        let bindings = vec![
            (VarId::new(0), Value::Int(1)),
            (VarId::new(1), Value::Int(2)),
            (VarId::new(2), Value::Int(3)),
        ];

        let env2 = env.extend_many(bindings);
        assert_eq!(env2.len(), 3);
        assert!(matches!(env2.lookup(VarId::new(1)), Some(Value::Int(2))));
    }

    #[test]
    fn test_env_shadowing() {
        let env = Env::new();
        let id = VarId::new(0);

        let env2 = env.extend(id, Value::Int(1));
        let env3 = env2.extend(id, Value::Int(2));

        assert!(matches!(env2.lookup(id), Some(Value::Int(1))));
        assert!(matches!(env3.lookup(id), Some(Value::Int(2))));
    }

    #[test]
    fn test_env_many_extensions_keep_every_binding() {
        // Far past the point where recent bindings are merged into the map.
        let mut env = Env::new();
        let mut snapshots = Vec::new();
        for i in 0..200 {
            env = env.extend(VarId::new(i % 150), Value::Int(i as i64));
            if i % 40 == 0 {
                snapshots.push((i, env.clone()));
            }
        }
        assert_eq!(env.len(), 150);
        assert_eq!(env.keys().count(), 150);
        // The newest binding of a variable wins, across a merge.
        assert!(matches!(env.lookup(VarId::new(10)), Some(Value::Int(160))));
        assert!(matches!(env.lookup(VarId::new(120)), Some(Value::Int(120))));
        // An earlier environment still sees what it saw.
        for (i, snapshot) in snapshots {
            let key = i % 150;
            assert_eq!(snapshot.len(), (i + 1).min(150));
            assert!(matches!(
                snapshot.lookup(VarId::new(key)),
                Some(Value::Int(v)) if *v == i as i64
            ));
            if i + 1 < 150 {
                assert!(snapshot.lookup(VarId::new(i + 1)).is_none());
            }
        }
        let many = Env::new().extend_many((0..100).map(|i| (VarId::new(i), Value::Int(1))));
        assert_eq!(many.len(), 100);
        assert!(!many.is_empty());
        assert!(Env::new().is_empty());
    }
}
