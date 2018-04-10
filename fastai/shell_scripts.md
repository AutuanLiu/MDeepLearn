```
   Description : shell notes
   Email : autuanliu@163.com
   Date：2018/04/09
```
- [Shell scripts](#shell-scripts)
    - [Notes](#notes)
    - [if-then](#if-then)
    - [case](#case)
    - [for](#for)
    - [while](#while)
    - [function](#function)
# Shell scripts
## Notes
```bash
$ echo This is a test
This is a test
$
```
* 默认情况下，不需要使用引号将要显示的文本字符串划定出来
* echo命令可用单引号或双引号来划定文本字符串
* 如果想把文本字符串和命令输出显示在同一行中，该怎么办呢？可以用echo语句的-n参数
* echo -n "The time and date are: "
* 你需要在字符串的两侧使用引号，保证要显示的字符串尾部有一个空格。命令输出将会在紧接着字符串结束的地方出现。
* 通过${variable}形式引用的变量。变量名两侧额外的花括号通常用来帮助识别美元符后的变量名
* 用户变量区分大小写
* 使用等号将值赋给用户变量。在变量、等号和值之间不能出现空格
* shell脚本会自动决定变量值的数据类型
* 在脚本的整个生命周期里， shell脚本中定义的变量会一直保持着它们的值，但在shell脚本结束时会被删除掉
* 引用一个变量值时需要使用美元符，而引用变量来对其进行赋值时则不要使用美元符
```bash
$ cat test4
#!/bin/bash
# assigning a variable value to another variable
value1=10
value2=$value1
echo The resulting value is $value2
$
```
* 有两种方法可以将命令输出赋给变量
    * 反引号字符（`）testing='date'
    * $()格式 testing=$(date)
    * shell会运行命令替换符号中的命令，并将其输出赋给变量testing
    * 赋值等号和命令替换字符之间没有空格
```bash
#!/bin/bash
# copy the /usr/bin directory listing to a log file
today=$(date +%y%m%d)
ls /usr/bin -al > log.$today
```
* +%y%m%d格式告诉date命令将日期显示为两位数的年月日的组合
* 命令替换会创建一个子shell来运行对应的命令
* 子shell（subshell）是由运行该脚本的shell所创建出来的一个独立的子shell（child shell）。正因如此，由该子shell所执行命令是无法使用脚本中所创建的变量的
* 在命令行提示符下使用路径./运行命令的话，也会创建出子shell；要是运行命令的时候不加入路径，就不会创建子shell
* 重定向可以用于输入，也可以用于输出，可以将文件重定向到命令输入
* 重定向输出 > , >>
* 输入重定向和输出重定向正好相反 <, << 输入重定向将文件的内容重定向到命令，而非将命令的输出重定向到文件 command < inputfile
* wc命令可以对对数据中的文本进行计数。默认情况下，它会输出3个值：
    * 文本的行数
    * 文本的词数
    * 文本的字节数
* 内联输入重定向符号是远小于号（<<）。除了这个符号，你必须指定一个文本标记来划分输入数据的开始和结尾。任何字符串都可作为文本标记，但在数据的开始和结尾文本标记必须一致。
```bash
$ command << marker
$ wc << EOF
```
* 将一个命令的输出作为另一个命令的输入 **管道 pipe**
```bash
$ rpm -qa > rpm.list
$ sort < rpm.list
$ rpm -qa | sort
$ $ rpm -qa | sort | more
$ command1 | command2
```

和命令替换所用的反引号（`）一样，管道符号在shell编程之外也很少用到。该符号由两个竖线构成，一个在另一个上面。然而管道符号的印刷体通常看起来更像是单个竖线
（|）。在美式键盘上，它通常和反斜线（\）位于同一个键。

不要以为由管道串起的两个命令会依次执行。 Linux系统实际上会同时运行这两个命令，在系统内部将它们连接起来。在第一个命令产生输出的同时，输出会被立即送给
第二个命令。数据传输不会用到任何中间文件或缓冲区

可以用一条文本分页命令（例如less或more）来强行将输出按屏显示

到目前为止，管道最流行的用法之一是将命令产生的大量输出通过管道传送给more命令。这对ls命令来说尤为常见

在bash中，在将一个数学运算结果赋给某个变量时，可以用美元符和方括号（$[ operation ]）将数学表达式围起来

```bash
$ var1=$[1 + 5]
$ echo $var1
6
$ var2=$[$var1 * 2]
$ echo $var2
12
$
```

* 在使用方括号来计算公式时，不用担心shell会误解乘号或其他符号
* bash shell数学运算符只支持整数运算。若要进行任何实际的数学计算，这是一个巨大的限制。
* bc 命令解决 shell 的浮点运算
* 第一部分options允许你设置变量。如果你需要不止一个变量，可以用分号将其分开。expression参数定义了通过bc执行的数学表达式
```bash
variable=$(echo "options; expression" | bc)
var1=$(echo "scale=4; 3.44 / 5" | bc)
echo The answer is $var1
$
```
将scale变量设置成了四位小数

```bash
$ cat test10
#!/bin/bash
var1=100
var2=45
var3=$(echo "scale=4; $var1 / $var2" | bc)
echo The answer for this is $var3
$
```
* bc命令能识别输入重定向，允许你将一个文件重定向到bc命令来处理。

shell中运行的每个命令都使用退出状态码（exit status）告诉shell它已经运行完毕。退出状态码是一个0～255的整数值，在命令结束运行时由命令传给shell。可以捕获这个值并在脚本中使用

Linux提供了一个专门的变量$?来保存上个已执行命令的退出状态码, 按照惯例，一个成功结束的命令的退出状态码是 0

exit命令允许你在脚本结束时指定一个退出状态码。

## if-then
```bash
if command
then
    commands
fi

if command; then
    commands
fi

if command
then
    commands
else
    commands
fi

if command1
then
    commands
elif command2
then
    more commands
fi
```

bash shell的if语句会运行if后面的那个命令。如果该命令的退出状态码（参见第11章）是0（该命令成功运行），位于then部分的命令就会被执行。如果该命令的退出状态码是
其他值， then 部分的命令就不会被执行， bash shell会继续执行脚本中的下一个命令。 fi语句用来表示if-then语句到此结束
* 在then部分，你可以使用不止一条命令
* 在elif语句中，紧跟其后的else语句属于elif代码块。它们并不属于之前的if-then代码块
* if 不能判断除了退出码之外的值

```bash
$ cat test1.sh
#!/bin/bash
# testing the if statement
if pwd
then
echo "It worked"
fi
$
```

test命令提供了在if-then语句中测试不同条件的途径。如果test命令中列出的条件成立，test命令就会退出并返回退出状态码 0
如果条件不成立， test命令就会退出并返回非零的退出状态码，这使得if-then语句不会再被执行

bash shell提供了另一种条件测试方法，无需在if-then语句中声明test命令

```bash
test condition

if test condition
then
    commands
fi

if [ condition ]
then
    commands
fi
```
* 第一个方括号之后和第二个方括号之前必须加上一个空格，否则就会报错
* test命令可以判断三类条件：
    * 数值比较
    * 字符串比较
    * 文件比较

name|含义
:---:|:---:
n1 -eq n2 | 检查n1是否与n2相等
n1 -ge n2 | 检查n1是否大于或等于n2
n1 -gt n2 | 检查n1是否大于n2
n1 -le n2 | 检查n1是否小于或等于n2
n1 -lt n2 | 检查n1是否小于n2
n1 -ne n2 | 检查n1是否不等于n2
str1 = str2 |检查str1是否和str2相同
str1 != str2 |检查str1是否和str2不同
str1 < str2 |检查str1是否比str2小
str1 > str2 |检查str1是否比str2大
-n str1 |检查str1的长度是否非0
-z str1 |检查str1的长度是否为0
-d file |检查file是否存在并是一个目录
-e file |检查file是否存在
-f file |检查file是否存在并是一个文件
-r file |检查file是否存在并可读
-s file |检查file是否存在并非空
-w file |检查file是否存在并可写
-x file |检查file是否存在并可执行
-O file |检查file是否存在并属当前用户所有
-G file |检查file是否存在并且默认组与当前用户相同
file1 -nt file2 |检查file1是否比file2新
file1 -ot file2 |检查file1是否比file2旧

```bash
if [ $value1 -gt 5 ]
then
echo "The test value $value1 is greater than 5"
fi
#
if [ $value1 -eq $value2 ]
then
echo "The values are equal"
else
echo "The values are different"
fi
#
$
```
* **bash shell只能处理整数**
* 大于号和小于号必须转义，否则shell会把它们当作重定向符号，把字符串值当作文件名；
* 大于和小于顺序和sort命令所采用的不同
```bash
val1=baseball
val2=hockey
#
if [ $val1 \> $val2 ]
then
echo "$val1 is greater than $val2"
else
echo "$val1 is less than $val2"
fi
```
* sort命令处理大写字母的方法刚好跟test命令相反

在比较测试中，大写字母被认为是小于小写字母的。但sort命令恰好相反。当你将同样的字符串放进文件中并用sort命令排序时，小写字母会先出现。
这是由各个命令使用的排序技术不同造成的。

如果你对数值使用了数学运算符号， shell会将它们当成字符串值，可能无法得到正确的结果

-n和-z可以检查一个变量是否含有数据

-e比较可用于文件和目录。要确定指定对象为文件，必须用-f比较

当-s比较成功时，说明文件中有数据

```bash
val1=testing
val2=''
#
if [ -n $val1 ]
then
echo "The string '$val1' is not empty"
else
echo "The string '$val1' is empty"
fi
#
if [ -z $val2 ]
then
echo "The string '$val2' is empty"
else
echo "The string '$val2' is not empty"
fi
#
if [ -z $val3 ]
then
echo "The string '$val3' is empty"
else
echo "The string '$val3' is not empty"
fi
```
```bash
if [ -d $jump_directory ]
if [ -e $location ]
if [ -f $item_name ]
if [ -r $pwfile ]
if [ -s $file_name ]
if [ -w $item_name ]
if [ -x test16.sh ]
if [ test19.sh -nt test18.sh ]
```

* if-then语句允许你使用布尔逻辑来组合测试
    * [ condition1 ] && [ condition2 ]
    * [ condition1 ] || [ condition2 ]

```bash
if [ -d $HOME ] && [ -w $HOME/testing ]
then
echo "The file exists and you can write to it"
else
echo "I cannot write to the file"
fi
```

* bash shell提供了两项可在if-then语句中使用的高级特性：
    * 用于数学表达式的双括号
    * 用于高级字符串处理功能的双方括号

符 号| 描 述
:---:|:---:
val++ |后增
val-- |后减
++val |先增
--val |先减
! |逻辑求反
~ |位求反
** |幂运算
<< |左位移
>> |右位移
& |位布尔和
| |位布尔或
&& |逻辑和
|| |逻辑或

```bash
if (( $val1 ** 2 > 90 ))
then
(( val2 = $val1 ** 2 ))
echo "The square of $val1 is $val2"
fi
```

* 不需要将双括号中表达式里的大于号转义。这是双括号命令提供的另一个高级特性
* 双方括号命令提供了针对字符串比较的高级特性

## case
```bash
case variable in
pattern1 | pattern2) commands1;;
pattern3) commands2;;
*) default commands;;
esac
```
* 可以通过竖线操作符在一行中分隔出多个模式模式。星号会捕获所有与已知模式不匹配的值。
* case命令会将指定的变量与不同模式进行比较。如果变量和模式是匹配的，那么shell会执行为该模式指定的命令

```bash
$ cat test26.sh
#!/bin/bash
# using the case command
#
case $USER in
rich | barbara)
echo "Welcome, $USER"
echo "Please enjoy your visit";;
testing)
echo "Special testing account";;
jessica)
echo "Do not forget to log off when you're done";;
*)
echo "Sorry, you are not allowed here";;
esac
$
$ ./test26.sh
Welcome, rich
Please enjoy your visit
$
```

## for

## while

## function
