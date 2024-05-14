# VPBK Project

## Zadání

* Úkolem je implementovat projekt deepface a napsat dokumentaci.
* Bude se kontrolovat funkčnost a způsob ověření funkčnosti
* Podle možností programu budou testovány buď natrénované/registrované tváře/osoby vůči novým obrázkům + vaše vlastní tvář (registrovat do systému a následně otestovat).


## Rozvržení projektu

    main.py                                     # Main Python File
    requirements.txt                            # Python App Requirements
    requirements-dev.txt                        # Python Developer Requirements
    ui/
        resources/                              # Folder with resources
                deepface-icon-labeled.png       # DeepFace Project Icon
        main_window.py                          # UI Main Window Python File
        main_window.ui                          # UI Main Window Qt Designer File
    mkdocs.yml                                  # The configuration file.
    docs/
        index.md                                # Homepage
        deepface.md                             # DeepFace Documentation
        implementace.md                         # Implementation Documentation
        code/
            deepfacehelper.md                   # Documentation for the DeepFaceHelper class.
            gui.md                              # Documentation for the GUI class.

## Zdroje

* [DeepFace](https://github.com/serengil/deepface)