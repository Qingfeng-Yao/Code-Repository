'''
Problem Discription:
You are given a phone number as a string number. number consists of digits, spaces ' ', and/or dashes '-'.
You would like to reformat the phone number in a certain manner. Firstly, remove all spaces and dashes. Then, group the digits from left to right into blocks of length 3 until there are 4 or fewer digits. The final digits are then grouped as follows:
2 digits: A single block of length 2.
3 digits: A single block of length 3.
4 digits: Two blocks of length 2 each.
The blocks are then joined by dashes. Notice that the reformatting process should never produce any blocks of length 1 and produce at most two blocks of length 2.
Return the phone number after formatting.

Example:
Input: number = "1-23-45 6"
Output: "123-456"
Explanation: The digits are "123456".
Step 1: There are more than 4 digits, so group the next 3 digits. The 1st block is "123".
Step 2: There are 3 digits remaining, so put them in a single block of length 3. The 2nd block is "456".
Joining the blocks gives "123-456".
'''

## self solution
### replace+分情况讨论/findall/按长度切分字符串
def reformatNumber(self, number: str) -> str:
    number = number.replace(" ", "").replace("-", "")
    if len(number)%3==0:
        return "-".join(re.findall(r".{3}", number))
    elif len(number)%3==2:
        return "-".join(re.findall(r".{3}", number[:-2])+[number[-2:]])
    elif len(number)%3==1:
        return "-".join(re.findall(r".{3}", number[:-4])+[number[-4:-2],number[-2:]])

## other solution
### 仅用正则匹配：插入
def reformatNumber(self, number: str) -> str:
    return re.sub('(...?(?=..))', r'\1-', re.sub('\D', '', number))