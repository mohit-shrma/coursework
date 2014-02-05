#!/usr/bin/python
import cgi
import cgitb; cgitb.enable()
import os

HTML_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><title>Form Details</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
</head><body>
<h1>%(MESSAGE)s</h1>
</body>
</html>"""

def parseForm ():
    form = cgi.FieldStorage()
    firstName = form['firstname'].value
    lastName = form['lastname'].value
    print HTML_TEMPLATE % {'MESSAGE' : 'Hello ' + firstName + ' ' + lastName}

print 'content-type: text/html\n'
parseForm()

