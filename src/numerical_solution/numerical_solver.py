"""
This script uses Ansys Fluent solver to solve a given case for different values of case parameters.
The case parameters are defined in the utils.parameters file.
Journal files are used to update the case parameters and run the case.
The solved case files are saved in the case_folder_path.
"""

# Importing necessary libraries
import ansys.fluent.core as pyfluent  # Python API for Ansys Fluent
import re  # Regular expressions for text substitution
import src.utils.parameters as parameters  # Custom module for parameter handling
import time  # For timing operations
from pathlib import Path  # For platform-independent path handling
from src.config import PARAMS_JOURNAL, RUN_JOURNAL, CASE_DIR, CASE_FILE, BASE_DIR  # Import path configurations
import src.utils.utils as utils
import os
import shutil

def next_unused_run_index(raw_root: str) -> int:
    """Smallest positive n such that ``Run_n`` is not an existing directory under ``raw_root``."""
    if not os.path.isdir(raw_root):
        return 1
    used: set[int] = set()
    for name in os.listdir(raw_root):
        path = os.path.join(raw_root, name)
        if not os.path.isdir(path):
            continue
        m = re.fullmatch(r"Run_(\d+)", name)
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return n

# Five input-parameter Definition lines (order: head, muscle, organ, T_amb, h)
_PARAM_DEFINITION_RE = re.compile(
    r'(cx-gui-do cx-set-expression-entry "Parameter Expression\*Table1\*ExpressionEntry3\(Definition\)" \'\(")([^"]+)("\s*\. 1\)\))'
)


def _replace_five_parameter_definitions(journal_text: str, five_values: list[str]) -> str:
    """Overwrite the five Definition expression strings in journal order; any previous values are fine."""
    if len(five_values) != 5:
        raise ValueError("five_values must have length 5")
    it = iter(five_values)

    def _sub(m):
        return m.group(1) + next(it) + m.group(3)

    new_text, n = _PARAM_DEFINITION_RE.subn(_sub, journal_text, count=5)
    if n != 5:
        raise ValueError(
            f"Expected 5 Parameter Expression Definition lines in journal, found {n}"
        )
    return new_text


def update_journal_files(params_journal_path, run_journal_path, sample_index: int, file_number):
    """
    Updates the Ansys Fluent journal files with new case parameters.

    Args:
        params_journal_path: Path to the journal file containing case parameters
        run_journal_path: Path to the journal file containing run instructions for the current case
        sample_index: Column index in ``parameters.dataset`` (which parameter set to write)
        file_number: Case index for the saved ``Exercise_Case_SteadyState_{file_number}.cas.h5`` name
    """
    five = parameters.expression_strings_for_sample(sample_index)
    params_path = Path(params_journal_path)
    file_contents = params_path.read_text(encoding="utf-8")
    file_contents = _replace_five_parameter_definitions(file_contents, five)
    params_path.write_text(file_contents, encoding="utf-8")

    run_path = Path(run_journal_path)
    file_contents = run_path.read_text(encoding="utf-8")

    new_filename = f"Exercise_Case_SteadyState_{file_number}.cas.h5"
    file_contents = re.sub(
        r"Exercise_Case_SteadyState_-?\d+\.cas\.h5",
        new_filename,
        file_contents,
    )

    run_path.write_text(file_contents, encoding="utf-8")

# start_time = time.time()

raw_root = os.path.join(BASE_DIR, "data", "raw")

# Get the next unused run index
n = next_unused_run_index(raw_root)

# Launch Ansys Fluent
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=4,
                                ui_mode="gui",
                                py=True)

# Read the initial case file
solver_session.settings.file.read_case(file_name=str(CASE_FILE))

# Update the journal files with new case parameters (sample 0)
update_journal_files(PARAMS_JOURNAL, RUN_JOURNAL, 0, 0)

# Read the updated journal files
solver_session.settings.file.read_journal(file_name_list=[str(PARAMS_JOURNAL), str(RUN_JOURNAL)])

# Exit Ansys Fluent
solver_session.exit()

# Organize the run
utils.organize_run(n)


case_files_dir = os.path.join(BASE_DIR, "data", "raw", f"Run_{n}", "case_files")
# Full case folder (libudf, journals, etc.) so Fluent can resolve UDF paths next to archived .h5.
shutil.copytree(CASE_DIR, case_files_dir, dirs_exist_ok=True)


# end_time = time.time()
# time_taken = end_time - start_time
# print(f"time_taken: {time_taken}")

# Run the case for all the parameters
for i in range(parameters.size_data-1):
    solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                            processor_count=4,
                                            ui_mode="gui",
                                            py=True)
    # use the highest run index 
    RUN_DIR = BASE_DIR / "data" / "raw" / f"Run_{n}"

    case_file = RUN_DIR / "case_files" / f"Exercise_Case_SteadyState_{i}.cas.h5"
    solver_session.settings.file.read_case(file_name=str(case_file))

    PARAMS_JOURNAL = RUN_DIR / "case_files" / "change_parameter.log"
    RUN_JOURNAL = RUN_DIR / "case_files" / "steady_and_transient.log"
    update_journal_files(PARAMS_JOURNAL, RUN_JOURNAL, i + 1, i + 1)
    solver_session.settings.file.read_journal(file_name_list=[str(PARAMS_JOURNAL), str(RUN_JOURNAL)])
    solver_session.exit()
    utils.organize_run(n)
