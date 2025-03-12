/* This file generated automatically. */
/*          Do not modify.            */
#include "udf.h"
#include "prop.h"
#include "dpm.h"
extern DEFINE_SOURCE(head_source,c,t,dS,eqn);
extern DEFINE_SOURCE(organ_source,c,t,dS,eqn);
extern DEFINE_SOURCE(muscle_source,c,t,dS,eqn);
extern DEFINE_EXECUTE_AT_END(blood_temperature);
extern DEFINE_PROFILE(convection_heat_transfer,t,i);
__declspec(dllexport) UDF_Data udf_data[] = {
{"head_source", (void(*)())head_source, UDF_TYPE_SOURCE},
{"organ_source", (void(*)())organ_source, UDF_TYPE_SOURCE},
{"muscle_source", (void(*)())muscle_source, UDF_TYPE_SOURCE},
{"blood_temperature", (void(*)())blood_temperature, UDF_TYPE_EXECUTE_AT_END},
{"convection_heat_transfer", (void(*)())convection_heat_transfer, UDF_TYPE_PROFILE},
};
__declspec(dllexport) int n_udf_data = sizeof(udf_data)/sizeof(UDF_Data);
#include "version.h"
__declspec(dllexport) void UDF_Inquire_Release(int *major, int *minor, int *revision)
{
  *major = RampantReleaseMajor;
  *minor = RampantReleaseMinor;
  *revision = RampantReleaseRevision;
}
