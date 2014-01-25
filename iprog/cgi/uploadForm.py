HTML_TEMPLATE = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html><head><title>File Upload</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
</head><body><h1>File Upload</h1>
<form action="uploadHandle.py" method="POST" enctype="multipart/form-data">
File name: <input name="file_input" type="file"><br>
<input name="submit" type="submit">
</form>
</body>
</html>"""


print HTML_TEMPLATE
