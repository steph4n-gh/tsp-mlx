import sys
import argparse

def patch_and_run(target_module_name: str, target_function: str):
    """
    Dynamically injects TSP into any MLX-based agent by monkey-patching mlx_lm.load.
    """
    print("\033[1;35m[\u03C4-Gate Injector] Initializing...\033[0m")
    
    try:
        import mlx_lm
    except ImportError:
        print("\033[1;31m[ERROR] mlx_lm not found. Is it installed in this environment?\033[0m")
        sys.exit(1)
        
    from tsp_mlx.sparse_cache import KVCacheManager
    from tsp_mlx.cortex_hook import CortexHook

    # 1. Save the original loader
    original_mlx_load = mlx_lm.load

    # 2. Define the Trojan Horse loader
    def tsp_patched_load(path_or_hf_repo: str, *args, **kwargs):
        print(f"\033[1;36m[\u03C4-Gate] Intercepting load request for: {path_or_hf_repo}\033[0m")
        
        # Let MLX load the weights from disk normally
        model, tokenizer = original_mlx_load(path_or_hf_repo, *args, **kwargs)
        
        # Silently attach the VRAM manager
        print("\033[1;36m[\u03C4-Gate] Attaching Neuro-Symbolic Paging Manager to model...\033[0m")
        hook = CortexHook(eval_interval=10, threshold=0.1)
        manager = KVCacheManager(hook, model=model, enable_compression=True, enable_consolidation=True)
        model._tsp_kv_manager = manager
        
        print("\033[1;32m[\u03C4-Gate] Model successfully patched. Returning control to host application.\033[0m")
        return model, tokenizer

    # 3. Apply the Monkey Patch globally
    mlx_lm.load = tsp_patched_load
    print("\033[1;32m[\u03C4-Gate Injector] mlx_lm.load has been successfully overridden.\033[0m")

    # 4. Boot the target agent
    print(f"\033[1;35m[\u03C4-Gate Injector] Booting target application: {target_module_name}.{target_function}()\033[0m\n")
    
    try:
        import importlib
        target_module = importlib.import_module(target_module_name)
        func = getattr(target_module, target_function)
        func()
    except Exception as e:
        print(f"\033[1;31m[ERROR] Failed to run target application: {e}\033[0m")

def cli_main():
    parser = argparse.ArgumentParser(description="Inject TSP into an MLX application.")
    parser.add_argument("--module", type=str, default="tsp_mlx.scripted_chat", help="Target module to run (e.g., gemini_cli.main)")
    parser.add_argument("--func", type=str, default="main", help="Target function to call (e.g., run)")
    args = parser.parse_args()
    
    # We need to make sure the tsp_mlx directory is in the path
    import os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    patch_and_run(args.module, args.func)

if __name__ == "__main__":
    cli_main()
