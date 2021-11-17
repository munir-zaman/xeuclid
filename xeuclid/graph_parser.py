Test = """
layout:
nodes:
A : $A$, 
B : $qB$; (1,1); color=RED; col=hello;

edges:
A -- B; color=Red;
A -> B;

end
"""

def text_is_tuple(text):
    text = text.strip()
    try:
        out = eval(f"type({text})==tuple")
    except:
        out = False
    return out


class GraphParse:

    def __init__(self, text: str):
        self.text = text
        self._parsed_text = self._parse_text() 
        self.layout = self._parsed_text["layout"]
        self.node_text = self._parsed_text["node_text"]
        self.edge_text = self._parsed_text["edge_text"]

    def _parse_text(self):
        text = self.text
        text_lines = [txt.strip() for txt in self.text.splitlines() if txt.strip()!=""]

        node_text_start = None
        node_text_end = None

        edge_text_start = node_text_end
        edge_text_end = None

        layout = ""

        for index in range(0, len(text_lines)):
            if text_lines[index].strip() == "nodes:" and node_text_start is None:
                node_text_start = index
            elif text_lines[index].strip() == "edges:" and node_text_end is None:
                node_text_end = index
                edge_text_start = index
            elif text_lines[index].strip() == "end" and edge_text_end is None:
                edge_text_end = index
            elif text_lines[index].strip().startswith("layout:") and layout == "":
                layout = text_lines[index].strip().split(":")[1]

        node_text = text_lines[node_text_start+1:node_text_end]
        edge_text = text_lines[edge_text_start+1:edge_text_end]

        return {"node_text": node_text, "edge_text": edge_text, "layout": layout}

    def _get_nodes(self):
        node_text = self.node_text
        node_dict = []

        for text in node_text:
            node_name, node_label = text.split(";")[0].split(":")
            node_name, node_label = node_name.strip(), node_label.strip()

            DICT_CONFIG = {}
            text_rest = text.split(";")[1::]
            if text_rest!=[] and not (len(text_rest)==1 and text_rest[0].strip()==""):
                for text_ in text_rest:
                    text_ = text_.strip()
                    if text_!="":
                        if not text_is_tuple(text_):
                            try:
                                kwarg, kwarg_value = text_.split("=")
                                if kwarg.strip()=="" or kwarg_value.strip()=="":
                                    raise SyntaxError(f"Could not parse line:\n----> {text}")
                                else:
                                    DICT_CONFIG.update({kwarg.strip() : kwarg_value.strip()})
                            except:
                                raise SyntaxError(f"Could not parse line:\n----> {text}")
                        else:
                            DICT_CONFIG.update({"pos": text_})
                    

            _Dict_ = {"name"    : node_name,
                      "label"   : node_label}

            _Dict_.update(DICT_CONFIG)

            node_dict.append(_Dict_)

        return node_dict

    def _get_edges(self):
        edge_text = self.edge_text
        edge_dict = []

        edge_types_ = ["--","->","<-", "<->"]

        for text in edge_text:
            text_0 = text.split(";")[0].strip()

            edge_type = None
            for edge_type_ in edge_types_:
                if len(text_0.split(edge_type_))==2:
                    edge_type = edge_type_
                    edge1 = text_0.split(edge_type_)[0].strip()
                    edge2 = text_0.split(edge_type_)[1].strip()

            if edge_type is None:
                raise SyntaxError(f"Could not parse line:\n----> {text}")

            DICT_CONFIG = {}
            text_rest = text.split(";")[1::]
            if text_rest!=[] and not (len(text_rest)==1 and text_rest[0].strip()==""):
                for text_ in text_rest:
                    text_ = text_.strip()
                    if text_!="":
                        try:
                            kwarg, kwarg_value = text_.split("=")
                            if kwarg.strip()=="" or kwarg_value.strip()=="":
                                raise SyntaxError(f"Could not parse line:\n----> {text}")
                            else:
                                DICT_CONFIG.update({kwarg.strip() : kwarg_value.strip()})
                        except:
                            raise SyntaxError(f"Could not parse line:\n----> {text}")


            _Dict_ = {"edge_1"      : edge1,
                      "edge_2"      : edge2,
                      "edge_type"   : edge_type}

            _Dict_.update(DICT_CONFIG)

            edge_dict.append(_Dict_)

        return edge_dict

print(GraphParse(Test)._get_edges())