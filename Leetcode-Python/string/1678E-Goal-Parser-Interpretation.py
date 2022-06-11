'''
Problem Discription:
You own a Goal Parser that can interpret a string command. The command consists of an alphabet of "G", "()" and/or "(al)" in some order. The Goal Parser will interpret "G" as the string "G", "()" as the string "o", and "(al)" as the string "al". The interpreted strings are then concatenated in the original order.
Given the string command, return the Goal Parser's interpretation of command.

Example:
Input: command = "G()(al)"
Output: "Goal"
'''

## self solution
### str.replace
def interpret(self, command: str) -> str:
    command = command.replace("()", "o")
    return command.replace("(al)", "al")

## other solution
### 减少中间步骤
def interpret(self, command: str) -> str:
    return command.replace("()", "o").replace("(al)", "al")
