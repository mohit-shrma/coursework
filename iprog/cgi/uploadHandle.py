#!/usr/bin/python
import cgi
import cgitb; cgitb.enable()
import os


UPLOAD_DIR = "/home/mohit/dev/github/coursework/iprog/tmp"
HTML_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><title>File Upload</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
</head><body><h1>File Upload</h1>
<h1>%(MESSAGE)s</h1>
</body>
</html>"""


def save_uploaded_file (form_field, upload_dir):
    form = cgi.FieldStorage()
    if not form.has_key(form_field): 
        print HTML_TEMPLATE % {'MESSAGE':"Error: file item not in form"}
        return
    fileitem = form[form_field]
    if not fileitem.file or len(fileitem.filename) ==0: 
        print HTML_TEMPLATE % {'MESSAGE': "Error: file not found"}
        return
    fout = file (os.path.join(upload_dir, fileitem.filename), 'wb')
    while 1:
        chunk = fileitem.file.read(100000)
        if not chunk: break
        fout.write (chunk)
    fout.close()
    print HTML_TEMPLATE % {'MESSAGE':'File uploaded successfully in ' + upload_dir}

print 'content-type: text/html\n'
save_uploaded_file ("file_input", UPLOAD_DIR)
