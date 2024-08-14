Get-ChildItem -Path "project-5_docs_helper\langchain-docs" -File | ForEach-Object {
    $newName = $_.BaseName + ".txt"
    Rename-Item -Path $_.FullName -NewName $newName
}