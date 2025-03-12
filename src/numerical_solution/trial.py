import ansys.fluent.core as pyfluent
solver_session = pyfluent.launch_fluent(precision=pyfluent.Precision.DOUBLE, 
                                           dimension=pyfluent.Dimension.THREE,
                                           processor_count=4)
    
# Read the case file
solver_session.file.read_case(file_name=r"C:\Users\Research\Desktop\CBT_pred\cases\exercise\with_sweating\case_1\Case1.cas.h5")
# solver_session.file.read_profile(file_name=r"C:\Users\Research\Desktop\CBT_pred\cases\exercise\with_sweating\case_1\profile.csv")
# m = solver_session.settings.setup.named_expressions.create(name="m")
# m.definition.set_state("profile('transient-temperature', 'temperature')")
# print(solver_session.settings.setup.named_expressions.compute(names=["m"]))
    
# solver = pyfluent.launch_fluent(mode=pyfluent.FluentMode.SOLVER)
# metab_head = solver.settings.setup.named_expressions["metab-head"]
# metab_head.definition.set_state("10 [W/m^3]")
metab_organ = solver_session.settings.setup.named_expressions["metab_organ"]
metab_organ.definition.set_state("10 [W/m^3]")
metab_organ.definition.print_state()

# print(metab_head())

 


































