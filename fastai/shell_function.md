```
   Description : shell function define
   Email : autuanliu@163.com
   Date：2018/04/10
```
- [function](#function)
    - [命令行参数](#%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0)
    - [shift](#shift)
    - [查找选项](#%E6%9F%A5%E6%89%BE%E9%80%89%E9%A1%B9)
    - [处理带值的选项](#%E5%A4%84%E7%90%86%E5%B8%A6%E5%80%BC%E7%9A%84%E9%80%89%E9%A1%B9)
    - [getopt](#getopt)
    - [getopts](#getopts)
    - [read](#read)
        - [从文件中读取](#%E4%BB%8E%E6%96%87%E4%BB%B6%E4%B8%AD%E8%AF%BB%E5%8F%96)
    - [文件描述符](#%E6%96%87%E4%BB%B6%E6%8F%8F%E8%BF%B0%E7%AC%A6)
    - [信号](#%E4%BF%A1%E5%8F%B7)
    - [function](#function)
        - [全局变量](#%E5%85%A8%E5%B1%80%E5%8F%98%E9%87%8F)
        - [局部变量](#%E5%B1%80%E9%83%A8%E5%8F%98%E9%87%8F)
        - [向函数传数组参数](#%E5%90%91%E5%87%BD%E6%95%B0%E4%BC%A0%E6%95%B0%E7%BB%84%E5%8F%82%E6%95%B0)
        - [从函数返回数组](#%E4%BB%8E%E5%87%BD%E6%95%B0%E8%BF%94%E5%9B%9E%E6%95%B0%E7%BB%84)
        - [函数递归](#%E5%87%BD%E6%95%B0%E9%80%92%E5%BD%92)
        - [创建库](#%E5%88%9B%E5%BB%BA%E5%BA%93)
# function

## 命令行参数

* 向shell脚本传递数据的最基本方法是使用命令行参数。命令行参数允许在运行脚本时向命令行添加数据
* bash shell会将一些称为位置参数（positional parameter）的特殊变量分配给输入到命令行中的所有参数

位置参数变量是标准的数字： $0是程序名， $1是第一个参数， $2是第二个参数，依次类推，直到第九个参数$9

```bash
factorial=1
for (( number = 1; number <= $1 ; number++ ))
do
factorial=$[ $factorial * $number ]
done
echo The factorial of $1 is $factorial
```

* 如果需要输入更多的命令行参数，则每个参数都必须用空格分开
* 可以在命令行上用文本字符串
* 要在参数值中包含空格，必须要用引号（单引号或双引号均可）
* 如果脚本需要的命令行参数不止9个，你仍然可以处理，但是需要稍微修改一下变量名。在第9个变量之后，你必须在变量数字周围加上花括号，比如${10}
* 可以用$0参数获取shell在命令行启动的脚本名

```bash
total=$[ ${10} * ${11} ]
echo The tenth parameter is ${10}
echo The eleventh parameter is ${11}
echo The total is $total
```

* basename命令会返回不包含路径的脚本名

```bash
$ cat test5b.sh
#!/bin/bash
# Using basename with the $0 parameter
#
name=$(basename $0)
echo
echo The script name is: $name
#
$ bash /home/Christine/test5b.sh
The script name is: test5b.sh
$
$ ./test5b.sh
The script name is: test5b.sh
```

* 在使用参数前一定要检查其中是否存在数据

```bash
$ cat test7.sh
#!/bin/bash
# testing parameters before use
#
if [ -n "$1" ]; then
    echo Hello $1, glad to meet you.
else
    echo "Sorry, you did not identify yourself. "
fi
```
* 特殊变量$#含有脚本运行时携带的命令行参数的个数。可以在脚本中任何地方使用这个特殊变量，就跟普通变量一样

```bash
echo There were $# parameters supplied.
```

* `$*` 和 `$@` 变量可以用来轻松访问所有的参数。这两个变量都能够在单个变量中存储所有的命令行参数

* `$*` 变量会将命令行上提供的所有参数当作一个单词保存。这个单词包含了命令行中出现的每一个参数值。基本上$*变量会将这些参数视为一个整体，而不是多个个体。
* `$@` 变量会将命令行上提供的所有参数当作同一字符串中的多个独立的单词。这样你就能够遍历所有的参数值，得到每个参数。这通常通过for命令完成

```bash
count=1
#
for param in "$*"
do
echo "\$* Parameter #$count = $param"
count=$[ $count + 1 ]
done
#
echo
count=1
#
for param in "$@"
do
echo "\$@ Parameter #$count = $param"
count=$[ $count + 1 ]
done
$
$ ./test12.sh rich barbara katie jessica
$* Parameter #1 = rich barbara katie jessica
$@ Parameter #1 = rich
$@ Parameter #2 = barbara
$@ Parameter #3 = katie
$@ Parameter #4 = jessica
$
```
* $*变量会将所有参数当成单个参数，而$@变量会单独处理每个参数。这是遍历命令行参数的一个绝妙方法

## shift
在使用shift命令时，默认情况下它会将每个参数变量向左移动一个位置。所以，变量$3的值会移到$2中，变量$2的值会移到$1中，而变量$1的值则会被删除（注意，变量$0的
值，也就是程序名，不会改变）

* 使用shift命令的时候要小心。如果某个参数被移出，它的值就被丢弃了，无法再恢复

```bash
echo "The original parameters: $*"
shift 2
echo "Here's the new first parameter: $1"
$
$ ./test14.sh 1 2 3 4 5
The original parameters: 1 2 3 4 5
Here's the new first parameter: 3
$
```
## 查找选项
```bash
while [ -n "$1" ]
do
case "$1" in
-a) echo "Found the -a option" ;;
-b) echo "Found the -b option" ;;
-c) echo "Found the -c option" ;;
*) echo "$1 is not an option" ;;
esac
shift
done
```

shell会用双破折线来表明选项列表结束。在双破折线之后，脚本就可以放心地将剩下的命令行参数当作参数，而不是选项来处理了

```bash
while [ -n "$1" ]
do
case "$1" in
-a) echo "Found the -a option" ;;
-b) echo "Found the -b option";;
-c) echo "Found the -c option" ;;
--) shift
break ;;
*) echo "$1 is not an option";;
esac
shift
done
#
count=1
for param in $@
do
echo "Parameter #$count: $param"
count=$[ $count + 1 ]
done
```
```bash
./test16.sh -c -a -b -- test1 test2 test3
```

## 处理带值的选项
## getopt
## getopts
## read
read命令从标准输入（键盘）或另一个文件描述符中接受输入 在收到输入后， read命令会将数据放进一个变量
```bash
echo -n "Enter your name: "
read name
echo "Hello $name, welcome to my program. "
#
$
$ ./test21.sh
Enter your name: Rich Blum
Hello Rich Blum, welcome to my program.
```

read命令包含了-p选项，允许你直接在read命令行指定提示符

-t选项指定了read命令等待输入的秒数
```bash
read -p "Please enter your age: " age
```
```bash
if read -t 5 -p "Please enter your name: " name
then
echo "Hello $name, welcome to my script"
else
echo
echo "Sorry, too slow! "
fi
```
### 从文件中读取
```bash
count=1
cat test | while read line
do
echo "Line $count: $line"
count=$[ $count + 1]
done
echo "Finished processing the file"
```
## 文件描述符
## 信号
* trap命令允许你来指定shell脚本要监看并从shell中拦截的Linux信号
* 当&符放到命令后时，它会将命令和bash shell分离开来，将命令作为系统中的一个独立的后台进程运行
* nice命令允许你设置命令启动时的调度优先级。要让命令以更低的优先级运行，只要用nice的-n命令行来指定新的优先级级别

## function
```bash
function func1 {
echo "This is an example of a function"
}
count=1
while [ $count -le 5 ]
do
func1
count=$[ $count + 1 ]
done
echo "This is the end of the loop"
func1
echo "Now this is the end of the script"
$
$ ./test1
This is an example of a function
This is an example of a function
```

* 函数定义不一定非得是shell脚本中首先要做的事，但一定要小心。如果在函数被定义前使用函数，你会收到一条错误消息
* 默认情况下，函数的退出状态码是函数中最后一条命令返回的退出状态码。在函数执行结束后，可以用标准变量$?来确定函数的退出状态码。

```bash
func1() {
echo "trying to display a non-existent file"
ls -l badfile
}
echo "testing the function: "
func1
```

* return命令来退出函数并返回特定的退出状态码

```bash
function dbl {
read -p "Enter a value: " value
echo "doubling the value"
return $[ $value * 2 ]
}
dbl
echo "The new value is $?"
```

* 函数可以使用标准的参数环境变量来表示命令行上传给函数的参数

```bash
function func7 {
echo $[ $1 * $2 ]
}
if [ $# -eq 2 ]
then
value=$(func7 $1 $2)
echo "The result is $value"
else
echo "Usage: badtest1 a b"
fi
$
$ ./test7
Usage: badtest1 a b
$ ./test7 10 15
The result is 150
```
### 全局变量
* 默认情况下，你在脚本中定义的任何变量都是全局变量。在函数外定义的变量可在函数内正常访问

### 局部变量
* 无需在函数中使用全局变量，函数内部使用的任何变量都可以被声明成局部变量
* 也可以在变量赋值语句中使用local关键字
* local关键字保证了变量只局限在该函数中。如果脚本中在该函数之外有同样名字的变量，那么shell将会保持这两个变量的值是分离的

```bash
function func1 {
local temp=$[ $value + 5 ]
result=$[ $temp * 2 ]
}
temp=4
value=6
func1
echo "The result is $result"
if [ $temp -gt $value ]
then
echo "temp is larger"
else
echo "temp is smaller"
fi
$
$ ./test9
The result is 22
temp is smaller
```

### 向函数传数组参数
* 将数组变量当作单个参数传递的话，它不会起作用
* 必须将该数组变量的值分解成单个的值，然后将这些值作为函数参数使用

```bash
function testit {
local newarray
newarray=(;'echo "$@"')
echo "The new array value is: ${newarray[*]}"
}
myarray=(1 2 3 4 5)
echo "The original array is ${myarray[*]}"
testit ${myarray[*]}
$
$ ./test10
The original array is 1 2 3 4 5
The new array value is: 1 2 3 4 5
```
### 从函数返回数组
```bash
function arraydblr {
local origarray
local newarray
local elements
local i
origarray=($(echo "$@"))
newarray=($(echo "$@"))
elements=$[ $# - 1 ]
for (( i = 0; i <= $elements; i++ ))
{
newarray[$i]=$[ ${origarray[$i]} * 2 ]
}
echo ${newarray[*]}
}
myarray=(1 2 3 4 5)
echo "The original array is: ${myarray[*]}"
arg1=$(echo ${myarray[*]})
result=($(arraydblr $arg1))
echo "The new array is: ${result[*]}"
$
$ ./test12
The original array is: 1 2 3 4 5
The new array is: 2 4 6 8 10
```
### 函数递归
```bash
function factorial {
if [ $1 -eq 1 ]
then
echo 1
else
local temp=$[ $1 - 1 ]
local result=$(factorial $temp)
echo $[ $result * $1 ]
fi
}
read -p "Enter value: " value
result=$(factorial $value)
echo "The factorial of $value is: $result"
$
$ ./test13
Enter value: 5
The factorial of 5 is: 120
```
### 创建库
* `!bash shell`允许创建函数库文件，然后在多个脚本中引用该库文件
* 这个过程的第一步是创建一个包含脚本中所需函数的公用库文件

```bash
#!/bin/bash
# using a library file the wrong way
./myfuncs
result=$(addem 10 15)
echo "The result is $result"
```
使用函数库的关键在于source命令。 source命令会在当前shell上下文中执行命令，而不是创建一个新shell。可以用source命令来在shell脚本中运行库文件脚本。这样脚本
就可以使用库中的函数了

source命令有个快捷的别名，称作点操作符（dot operator）。要在shell脚本中运行myfuncs库文件，只需添加下面这行

```bash
. ./myfuncs
```
```bash
#!/bin/bash
# using functions defined in a library file
. ./myfuncs
value1=10
value2=5
result1=$(addem $value1 $value2)
result2=$(multem $value1 $value2)
result3=$(divem $value1 $value2)
echo "The result of adding them is: $result1"
echo "The result of multiplying them is: $result2"
echo "The result of dividing them is: $result3"
```

一个非常简单的方法是将函数定义在一个特定的位置，这个位置在每次启动一个新shell的时
候，都会由shell重新载入。最佳地点就是.bashrc文件

只要是在shell脚本中，都可以用source命令（或者它的别名点操作符）将库文件中的函数添加到你的.bashrc脚本中