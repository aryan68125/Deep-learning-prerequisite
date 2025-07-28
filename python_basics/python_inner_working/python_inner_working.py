print(f"Test program")

from utility_import_test import TestClass

class Main(TestClass):
    def driver_fun(self):
        result = self.print_welcome_message("Aditya")
        print(result)

if __name__ == "__main__":
    obj = Main()
    obj.driver_fun()