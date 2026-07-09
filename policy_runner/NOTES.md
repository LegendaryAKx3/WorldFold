# Policy Runner Usage

```
python scripts/run_policy.py --env MuJoCoTouch-v1 --policy random --episodes 5
```

## Adding a new policy later

1. Create `policy_runner/policies/my_policy.py`:

   ```python
   from policy_runner.policy import Policy

   class MyPolicy(Policy):
       def reset(self):
           ...  # optional: reset any episode-local state

       def act(self, observation):
           ...  # return an action from self.action_space
   ```

2. Register it in `policy_runner/loader.py`:

   ```python
   from policy_runner.policies.my_policy import MyPolicy

   POLICY_REGISTRY = {
       "random": RandomPolicy,
       "my_policy": MyPolicy,
   }
   ```

3. Run it with no other changes:

   ```
   python scripts/run_policy.py --env MuJoCoTouch-v1 --policy my_policy
   ```

