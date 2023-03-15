$versions=("3.6.8", "3.7.7", "3.8.3", "3.9.4", "3.10.1")

foreach($ver in $versions)
{
    $download_url="https://www.python.org/ftp/python/${ver}/python-${ver}-amd64.exe"
    $download_file="python-${ver}-amd64.exe"
    echo "Download the python-${ver} from ${download_url}"
    curl.exe -SL $download_url --output $download_file
    if ($LASTEXITCODE -ne 0) {
        echo "Download file ${download_file} failed"
    }
    $process = Start-Process "python-${ver}-amd64.exe" -ArgumentList @("/quiet","Include_launcher=0", "TargetDir=$PWD\python_dev\$ver", "Shortcuts=0", "InstallLauncherAllUsers=0") -Wait -PassThru
    $EXITCODE=$process.ExitCode
    if($EXITCODE -eq 0)
    {
        cp $PWD/python_dev/$ver/python.exe $PWD/python_dev/$ver/python3.exe
    }
    else {
        echo "Setup python $ver failed"
    }  
    del $download_file
}