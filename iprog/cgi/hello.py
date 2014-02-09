#!/usr/bin/python
import cgi
import cgitb; cgitb.enable()
import os


def parseForm ():
    form = cgi.FieldStorage()
    firstName = form['firstname'].value
    lastName = form['lastname'].value
    print HTML_RESP_TEMPLATE % {'MESSAGE' : 'Hello ' + firstName + ' ' + lastName}


HTML_RESP_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><title>Form Details</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
</head><body>
<h1>%(MESSAGE)s</h1>
</body>
</html>"""


HTML_FORM_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><title>Form</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
</head><body><h1>Enter your details</h1>
<form action="helloHandle.py" method="POST" enctype="multipart/form-data">
First name: <input name="firstname" type="text"></br>
Last name: <input name="lastname" type="text">  
<input name="submit" type="submit">
</form>
</body>
</html>"""

print 'content-type: text/html\n'

form = cgi.FieldStorage()
if 'firstname' in form and len(form['firstname'].value) > 0:
    parseForm()
else:
    print HTML_FORM_TEMPLATE

