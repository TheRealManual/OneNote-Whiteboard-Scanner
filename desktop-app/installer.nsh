; Custom NSIS installer script
; Don't install dependencies during setup - let the app do it on first run
; This avoids permission issues during installation

!macro customInstall
  ; Just show a message that first run will set up dependencies
  DetailPrint "Backend dependencies will be installed on first app launch"
!macroend
