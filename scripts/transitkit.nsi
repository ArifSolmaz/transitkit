; NSIS installer script for TransitKit

!include "MUI2.nsh"

; Basic settings
Name "TransitKit"
OutFile "TransitKit_Setup.exe"
InstallDir "$PROGRAMFILES\TransitKit"
RequestExecutionLevel admin

; Interface settings
!define MUI_ABORTWARNING

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Sections
Section "TransitKit" SecMain
  SetOutPath "$INSTDIR"
  
  ; Copy files
  File /r "dist\transitkit.exe"
  File "LICENSE"
  File "README.md"
  
  ; Create start menu shortcut
  CreateDirectory "$SMPROGRAMS\TransitKit"
  CreateShortcut "$SMPROGRAMS\TransitKit\TransitKit.lnk" "$INSTDIR\transitkit.exe"
  CreateShortcut "$SMPROGRAMS\TransitKit\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\TransitKit" \
                   "DisplayName" "TransitKit"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\TransitKit" \
                   "UninstallString" '"$INSTDIR\uninstall.exe"'
SectionEnd

; Uninstaller
Section "Uninstall"
  ; Remove files
  Delete "$INSTDIR\*"
  RMDir /r "$INSTDIR"
  
  ; Remove shortcuts
  Delete "$SMPROGRAMS\TransitKit\*"
  RMDir "$SMPROGRAMS\TransitKit"
  
  ; Remove registry entries
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\TransitKit"
SectionEnd