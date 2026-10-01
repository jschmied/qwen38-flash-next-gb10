A third commit, [`5824347`](https://github.com/jschmied/TensorFold/commit/5824347c88fc387fdff6b13399d6bb4ef941c400), fixes two paths the first two missed. Each new lone request with a different prompt still recaptured every graph, because the previous request's kept prompt end held the slot. `copy_from` could also fail on sizes once the slot kept its rows. With four lone requests on `--parallel 4` there are now 0 captures each (was 33/32/30/19), and every reply is byte-identical. Details are in the description.

Written with AI assistance (Claude Code).
