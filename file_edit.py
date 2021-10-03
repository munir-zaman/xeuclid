import os

def create_file(file_name):
    with open(file_name,"x") as file:
        pass

def write_to_file(file_name,text):
    with open(file_name,"a") as file:
        file.write(text+"\n")

def readlines_file(file_name):
    with open(file_name, "r") as file:
        out = file.readlines()
    return out

class File():

    def __init__(self, name):
        self.name = name

    def create(self):
        create_file(self.name)

    def delete(self):
        os.system(f"del {self.name}")

    def read(self):
        with open(self.name, "r") as file:
            out = file.read()
        return out

    def readlines(self):
        with open(self.name, "r") as file:
            out = file.readlines()
        return out

    def append(self, text):
        write_to_file(self.name, text)

    def rewrite(self, text):

        if type(text) == str: 
            with open(self.name, "w+") as file:
                file.write(text)
        elif type(text) == list:
            with open(self.name, "w+") as file:
                file.writelines(text)

    def replace(self, lineno, new_text):
        lineno -= 1
        file_content = self.readlines()
        file_length = len(file_content)
        text = new_text
        if 0 <= lineno <= file_length:
            file_content[lineno] = new_text
        else:
            print("Line no out of range.")

        self.rewrite(file_content)

    def write(self, text, lineno = None):
        content = self.readlines()
        lineno -= 1

        if lineno is None:
            self.append(text)

        elif type(lineno) == int and 0 <= lineno <= len(content) - 1:
            new_content = content[:lineno] + [text] + content[lineno:]
            self.rewrite(new_content)

        elif type(lineno) == int and not (0 <= lineno <= len(content) - 1):
            new_content = content[:] + (lineno - len(content))*["\n"] + [text]
            self.rewrite(new_content)


