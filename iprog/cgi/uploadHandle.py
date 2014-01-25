#!/usr/bin/python
import cgi
import cgitb; cgitb.enable()

UPLOAD_DIR = ""

def save_uploaded_file (form_field, upload_dir):
    form = cgi.FieldStorage()
    if not form.has_key(form_field): 
        print "Error: file item not in form"
        return
    fileitem = form[form_field]
    if not fileitem.file: 
        print "Error: file not found"
        return
    fout = file (os.path.join(upload_dir, fileitem.filename), 'wb')
    while 1:
        chunk = fileitem.file.read(100000)
        if not chunk: break
        fout.write (chunk)
    fout.close()
    print 'File uploaded successfully in ' + upload_dir

save_uploaded_file ("file_input", UPLOAD_DIR)
