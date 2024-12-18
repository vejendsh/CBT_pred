import ansys.fluent.core as pyfluent

solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, dimension=pyfluent.Dimension.THREE,
                                processor_count=2,
                                ui_mode="gui",
                                py=True)
# input("Fluent has launched successfully!")
solver_session.tui.file.read_case(file_name=r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\Exercise_Case_SteadyState.cas.h5")
solver_session.tui.file.read_journal(path=r"C:\Users\vejendsh\OneDrive - University of Cincinnati\Swarup_MS_Thesis_2015\Wholebodymodel\Exercise\With_sweating\Case1\steady_and_transient.log")

