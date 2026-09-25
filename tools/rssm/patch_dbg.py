import sys, os
p = os.path.join(sys.argv[1], "model_executor/layers/mamba/abstract.py"); s = open(p).read()
if "FNRSSMDBG" not in s:
    old = "        self.kv_cache = tuple(states)\n"
    assert s.count(old) == 1
    s = s.replace(old, old + """        if hasattr(self, "use_gdn_recoverssm") and not getattr(type(self), "_fnrssm_dbg_done", False):  # FNRSSMDBG
            type(self)._fnrssm_dbg_done = True
            from vllm.logger import init_logger as _il
            _il(__name__).warning("FNRSSMDBG bind: %d states, shapes %s, dtypes %s, page bytes %d, flag %s, num_spec %s",
                                  len(states), list(self.get_state_shape()), list(self.get_state_dtype()),
                                  pages.shape[1], getattr(self.cache_config, "use_kda_recoverssm", None),
                                  getattr(self, "num_spec", None))
""")
    open(p, "w").write(s); print("patched dbg")
