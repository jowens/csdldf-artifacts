"c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
# set PATH=c:\Users\jdowens\Documents\working\dawn-latest\install\RelWithDebInfo\bin;%PATH%
cmake -S . -B out/RelWithDebInfo -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=c:/Users/jdowens/Documents/working/dawn-latest/install/RelWithDebInfo
cmake --build out/RelWithDebInfo
# command-line: .\out\RelWithDebInfo\dawn.exe csdldf_prof