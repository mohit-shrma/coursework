#!/usr/bin/python
HTML_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
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

print "content-type: text/html\n"
print HTML_TEMPLATE
