tplt = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ResultPreview</title>
</head>
<style>
    body {
        padding: 0 1%;
        font-size: 1.2em;
    }

    p {
        margin: 0.25em 0;
    }

    del {
        text-decoration: line-through;
    }

    code.inline {
        padding: 1px 3px;
        font-size: .72rem;
        line-height: .72rem;
        color: #c25;
        background-color: #f7f7f9;
        -webkit-border-radius: 3px;
        -moz-border-radius: 3px;
        border-radius: 3px;
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        border: 1px solid #e1e2e8;
    }
</style>
<body>
%REPLACE%
</body>
</html>
"""