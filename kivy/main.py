import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput

kivy.require("1.10.1")

#epicapp inheriting from app
class ConnectPage(GridLayout):
    def __init__(self,**kwargs):

        super().__init__(**kwargs)

        self.cols = 2
        self.add_widget(Label(text = "IP:"))
        self.ip = TextInput(multiline = True)
        self.add_widget(self.ip)

        self.add_widget(Label(text = "Port:"))
        self.port = TextInput(multiline = False)
        self.add_widget(self.port)

        self.add_widget(Label(text = "UserName:"))
        self.username = TextInput(multiline = False)
        self.add_widget(self.username)


class EpicApp(App):
    def build(self):
        return ConnectPage()

if __name__ == "__main__":
    EpicApp().run()
