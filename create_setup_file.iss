#define MyAppVersion "1.01"

[Setup]
AppName=Synapto Catcher
AppVersion={#MyAppVersion}
DefaultDirName={pf}\Synapto Catcher
DefaultGroupName=Synapto Catcher
OutputDir=setup
OutputBaseFilename=synapto_catcher_setup_{#MyAppVersion}
SetupIconFile=.\images\synaptocatcher.ico
WizardImageFile=.\images\synaptocatcher_logo.bmp
WizardSmallImageFile=.\images\synaptocatcher_small.bmp

[Files]
Source: ".\output\synapto catcher\synapto catcher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: ".\output\synapto catcher\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Synapto Catcher"; Filename: "{app}\synapto catcher.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall Synapto Catcher"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Icons]
Name: "{group}\Synapto Catcher"; Filename: "{app}\synapto catcher.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall Synapto Catcher"; Filename: "{uninstallexe}"
Name: "{commondesktop}\Synapto Catcher"; Filename: "{app}\synapto catcher.exe"; Tasks: desktopicon; WorkingDir: "{app}"