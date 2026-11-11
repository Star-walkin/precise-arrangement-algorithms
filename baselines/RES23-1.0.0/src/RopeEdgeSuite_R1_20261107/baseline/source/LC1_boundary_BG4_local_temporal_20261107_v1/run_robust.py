"""Entry point for a fresh spatial + reference camera + robust temporal run."""
import run_bg4
from temporal_robust import Counter,PARAMETERS
run_bg4.Counter=Counter
run_bg4.TEMPORAL_PARAMETERS=PARAMETERS
if __name__=='__main__':run_bg4.main()
