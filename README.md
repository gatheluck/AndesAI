### Test
Use `scripts/test.py` to test the network. `scripts/experiments/test_multi.py` is able to test the robustness against multiple size of epsilon at same time.
```bash
# Example usage (robustness test against specific attack method, norm, eps)
cd scripts 
python test.py  -t ${path_to_target} -a resnet50 -d imagenet100 --attack pgd --attack_norm linf --attack_eps 8

# Example usage (robustness test against specific attack method, norm and multiple eps)
cd scripts
python experiments/test_multi.py  -t ${path_to_target} -a resnet50 -d imagenet100 --attack pgd --attack_norm linf
```

We also provide cript generator for training by [ABCI](https://abci.ai/) which is the world's first large-scale open AI computing infrastructure.
Use `scripts/experiments/train_binary_abci.py` to generate shell scripts for ABCI. 
If user specify cost, at_norm, at_eps, script generator makes only scripts for training on such values.
If the values are not specified, the generator takes for loop in some range of values.   
Example usage:
```bash
# Example usage
cd scripts
python experiments/train_binary_abci.py -d cifar10 -l ../logs/train --script_root ../logs/abci_script --run_dir . --abci_log_dir ../logs/abci_log --user ${your_abci_user_id} --env ${abci_conda_environment} --cost 0.3