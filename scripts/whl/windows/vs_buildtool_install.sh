#Install Visual Studio Build Tools
#Reference: https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019
#Component IDS:https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2019
echo "Try to download the setup file from https://aka.ms/vs/16/release/vs_buildtools.exe"
curl -SL https://aka.ms/vs/16/release/vs_buildtools.exe --output ./vs_buildtools.exe
./vs_buildtools.exe --installPath $PWD/vs --nocache --wait --quiet --norestart  \
                    --add Microsoft.Component.MSBuild \
                    --add Microsoft.VisualStudio.Component.Roslyn.Compiler \
                    --add Microsoft.VisualStudio.Component.Windows10SDK.18362 \
                    --add Microsoft.VisualStudio.Workload.VCTools \
                    --add Microsoft.VisualStudio.Component.TextTemplating \
                    --add Microsoft.VisualStudio.Component.VC.CoreIde \
                    --add Microsoft.VisualStudio.Component.VC.Redist.14.Latest \
                    --add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core \
                    --add Microsoft.VisualStudio.Component.VC.CMake.Project \
                    --add Microsoft.VisualStudio.Component.VC.14.26.x86.x64
rm vs_buildtools.exe

if [[ $ERRORLEVEL -ne 3010 ]]; then
    echo "Error exit code:" $ERRORLEVEL
    curl.exe -o vscollect.exe -SL "https://aka.ms/vscollect.exe"
    ./vscollect.exe -Wait -PassThru -zip ${PWD}/log.zip
fi