#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#include <stdarg.h>
#include <iostream>


#include "rpc/client.h"

#include "fmi2Functions.h"


using namespace std;

#ifdef _WIN32
template<typename T> T *get(HMODULE libraryHandle, const char *functionName) {
    void *fp = GetProcAddress(libraryHandle, functionName);
	return reinterpret_cast<T *>(fp);
}
#else
template<typename T> T *get(void *libraryHandle, const char *functionName) {
    void *fp = dlsym(libraryHandle, functionName);
    cout << functionName << " = " << fp << endl;
    return reinterpret_cast<T *>(fp);
}
# endif

void logger(fmi2ComponentEnvironment componentEnvironment, fmi2String instanceName, fmi2Status status, fmi2String category, fmi2String message, ...) {
    
    printf("[%d][%s] ", status, instanceName);

    va_list args;
    va_start(args, message);

    vprintf(message, args);

    va_end(args);

    printf("\n");
}

#define CALL(f) if ((status = f) != fmi2OK) goto out;



extern string server_ip;


int main(int argc, char *argv[]) {

   if ( argc !=2 ){
       printf("\nError!  Usage:  %s serverip \n\n ", argv[0]);
       exit(-1);
   }

   server_ip = argv[1];

   //char cmd[512];
   //snprintf(cmd, 512, "ssh %s echo",server_ip.data());
   //if ( system(cmd) !=0 ){
   //    printf("Need to have access to remote server !, exit -1 \n");
   //    return(-1);
   //}
   // dummy code
   // now to copy bouncingBall.so to server and start

  //rpc::client rpclient("localhost", 8080);
  //string input, result;
  //while (std::getline(std::cin, input)) {
  //  if (!input.empty()) {
  //    result = rpclient.call("echo", input).as<string>();
  //    std::cout << result << std::endl;
  //  }
  //}
  //exit(0);

  fmi2CallbackFunctions functions = { logger, nullptr, nullptr, nullptr, nullptr };

  auto c = fmi2Instantiate("bb", fmi2CoSimulation, "{8c4e810f-3df3-4a00-8276-176fa3c9f003}", "", &functions, fmi2False, fmi2False);
 

  if (!c) {
	  cout << "Failed to instantiate FMU." << endl;
	  return 1;
  }

  fmi2Status status = fmi2OK;
  
  const fmi2Real stopTime = 1;
  const fmi2Real stepSize = 0.1;
  
  const fmi2ValueReference vr[2] = { 0, 2 };
  fmi2Real value[2] = { 0, 0 };
  
  fmi2Real time = 0;
  
  fmi2SetupExperiment(c, fmi2False, 0, 0, fmi2True, stopTime);
  fmi2EnterInitializationMode(c);
  fmi2ExitInitializationMode(c);



  const fmi2ValueReference vr11[1] = { 5 };
  fmi2Real value11[1] = { 0.5 };
  fmi2SetReal(c, vr11, 1, value11);

   while (time <= stopTime) {
        fmi2GetReal(c, vr, 2, value);
		cout << time << ", " << value[0] << ", " << value[1] << endl;
        fmi2DoStep(c, time, stepSize, fmi2True);
		time += stepSize;
	}

	fmi2Terminate(c);
	
	fmi2FreeInstance(c);

  cout << "pass to final step" << endl;


  return 0;

}


/*
    if (argc != 2) {
        cout << "Usage: client_test <library_path>" << endl;
        return 1;
    }

    const char *libraryPath = argv[1];
   
	// load the shared library
# ifdef _WIN32
	auto l = LoadLibraryA(libraryPath);
# else
	auto l = dlopen(libraryPath, RTLD_LAZY);
# endif

    if (!l) {
        cout << "Failed to load shared library." << endl;
        return 1;
    }

	auto getTypesPlatform        = get<fmi2GetVersionTYPE>              (l, "fmi2GetTypesPlatform");
    auto getVersion              = get<fmi2GetVersionTYPE>              (l, "fmi2GetVersion");
	auto instantiate             = get<fmi2InstantiateTYPE>             (l, "fmi2Instantiate");
	auto setupExperiment         = get<fmi2SetupExperimentTYPE>         (l, "fmi2SetupExperiment");
	auto enterInitializationMode = get<fmi2EnterInitializationModeTYPE> (l, "fmi2EnterInitializationMode");
	auto exitInitializationMode  = get<fmi2ExitInitializationModeTYPE>  (l, "fmi2ExitInitializationMode");
	auto getReal                 = get<fmi2GetRealTYPE>                 (l, "fmi2GetReal");
	auto doStep                  = get<fmi2DoStepTYPE>                  (l, "fmi2DoStep");
	auto terminate               = get<fmi2TerminateTYPE>               (l, "fmi2Terminate");
	auto freeInstance            = get<fmi2FreeInstanceTYPE>            (l, "fmi2FreeInstance");

	auto typesPlatform = getTypesPlatform();

    cout << "Types Platform: " << typesPlatform << endl;

	auto version = getVersion();
    
    cout << "FMI Version: " << version << endl;

    fmi2CallbackFunctions functions = { logger,	nullptr, nullptr, nullptr, nullptr };

	auto c = instantiate("bb", fmi2CoSimulation, "{8c4e810f-3df3-4a00-8276-176fa3c9f003}", "", &functions, fmi2False, fmi2False);

    if (!c) {
        cout << "Failed to instantiate FMU." << endl;
        return 1;
    }

	fmi2Status status = fmi2OK;

	const fmi2Real stopTime = 1;
	const fmi2Real stepSize = 0.1;

    const fmi2ValueReference vr[2] = { 1, 3 };
    fmi2Real value[2] = { 0, 0 };

    fmi2Real time = 0;

	CALL(setupExperiment(c, fmi2False, 0, 0, fmi2True, stopTime));

    CALL(enterInitializationMode(c));
    CALL(exitInitializationMode(c));

	while (time <= stopTime) {
        CALL(getReal(c, vr, 2, value));
		cout << time << ", " << value[0] << ", " << value[1] << endl;
        CALL(doStep(c, time, stepSize, fmi2True));
		time += stepSize;
	}

	CALL(terminate(c));
	
	freeInstance(c);

out:

#ifdef _WIN32
	FreeLibrary(l);
#else
    dlclose(l);
#endif

	return status;
}

*/
