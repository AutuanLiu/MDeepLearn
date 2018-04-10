```
   Description : shell notes
   Email : autuanliu@163.com
   Date：2018/04/09
```
- [Shell Notes](#shell-notes)
    - [Notes](#notes)
    - [Command](#command)

# Shell Notes
## Notes
* `man` 命令用来访问存储在 Linux 系统上的手册页面。在想要查找的工具的名称前面输入 `man` 命令，就可以找到那个工具相应的手册条目
* Linux会在根驱动器上创建一些特别的目录，我们称之为挂载点（mount point）。挂载点是虚拟目录中用于分配额外存储设备的目录。虚拟目录会让文件和目录出现在这些挂载点目录中，然而实际上它们却存储在另外一个驱动器中。
* 系统文件会存储在根驱动器中，而用户文件则存储在另一驱动器中
* ls命令输出的列表是按字母排序的（按列排序而不是按行排序）
* ls -F -F参数的ls命令轻松区分文件和目录, -F 参数在目录名后加了正斜线（/），以方便用户在输出中分辨它们。在可执行文件（比如上面的my_script文件）的后面加个星号，以便用户找出可在系统上运行的文件。
* Linux经常采用隐藏文件来保存配置信息。在Linux上，隐藏文件通常是文件名以点号开始的文件。
* 要把隐藏文件和普通文件及目录一起显示出来，就得用到 -a 参数
* -R 参数是ls命令可用的另一个参数，叫作递归选项。它列出了当前目录下包含的子目录中的文件。
* 命令的参数可以合并
* drwxr-xr-x 2 christine christine 4096 Apr 22 20:37 Desktop
    * 目录（d）、文件（-）、字符型文件（c）或块设备（b）
    * 文件的权限
    * 文件的硬链接总数
    * 文件属主的用户名
    * 文件属组的组名
    * 文件的大小（以字节为单位）
    * 文件的上次修改时间
    * 文件名或目录名
* ls -l my_script
    * 问号（?）代表一个字符 ls -l my_scr?pt
    * 星号（*）代表零个或多个字符
    * 中括号以及在特定位置上可能出现的两种字符： a或i ls -l my_scr[ai]pt
    * 可以使用感叹号（!）将不需要的内容排除在外 ls -l f[!a]ll
* cp 最好是加上-i选项，强制shell询问是否需要覆盖已有文件。
* 目录的话**一定在结尾加上 /**
* 链接是目录中指向文件真实位置的占位符
    * 符号链接 -- 实实在在的文件，它指向存放在虚拟目录结构中某个地方的另一个文件。这两个通过符号链接在一起的文件，彼此的内容并不相同。
    * 硬链接 -- 硬链接会创建独立的虚拟文件，其中包含了原始文件的信息及位置。但是它们从根本上而言是同一个文件
* 显示在长列表中符号文件名后的->符号表明该文件是链接到文件data_file上的一个符号链接。
* ls -i *data_file 产看文件的 inode 文件由 iNode 唯一标识
    ```bash
    -rw-rw-r-- 1 christine christine 1092 May 21 17:27 data_file
    lrwxrwxrwx 1 christine christine 9 May 21 17:29 sl_data_file -> data_file
    ```
* 带有硬链接的文件共享inode编号。这是因为它们终归是同一个文件。还要注意的是，链接计数（列表中第三项）显示这两个文件都有两个链接。另外，它们的文件大小也一模一样。
    ```bash
    296892 -rw-rw-r-- 2 christine christine 189 May 21 17:56 code_file
    296892 -rw-rw-r-- 2 christine christine 189 May 21 17:56 hl_code_file
    ```
* 只能对处于同一存储媒体的文件创建硬链接。要想在不同存储媒体的文件之间创建链接，只能使用符号链接。
* 同一个文件拥有多个链接，这完全没有问题。但是，千万别创建软链接文件的软链接
* mv命令可以将文件和目录移动到另一个位置或重新命名。
* 移动文件会将文件名从fall更改为fzll，但inode编号和时间戳保持不变。这是因为mv只影响文件名。
    * 也可以使用 mv 来移动文件的位置。
* 和cp命令类似，也可以在mv命令中使用-i参数。这样在命令试图覆盖已有的文件时，你就会得到提示。
* 使用mv命令移动文件位置并修改文件名称，这些操作只需一步就能完成。
```bash
mv /home/christine/Pictures/fzll /home/christine/fall
```
* -i命令参数提示你是不是要真的删除该文件。 bash shell中没有回收站或垃圾箱，文件一旦删除，就无法再找回。因此，在使用`rm`命令时，要养成总是加入`-i`参数的好习惯
* 要想同时创建多个目录和子目录，需要加入-p参数
* 默认情况下， rmdir命令只删除空目录
* 也可以在整个非空目录上使用rm命令。使用-r选项使得命令可以向下进入目录，删除其中的文件，然后再删除目录本身
    * rm -ri My_Dir
    * rm -rf Small_Dir
* file命令是一个随手可得的便捷工具。 它能够探测文件的内部，并决定文件是什么类型的
* file命令不仅能确定文件中包含的文本信息，还能确定该文本文件的字符编码， ASCII
* 可以使用file命令作为另一种区分目录的方法
* file命令能够确定该程序编译时所面向的平台以及需要何种类型的库
```bash
$ file /bin/ls
/bin/ls: ELF 64-bit LSB executable, x86-64, version 1 (SYSV),
dynamically linked (uses shared libs), for GNU/Linux 2.6.24,
[...]
```
* cat
    * 只想给有文本的行加上行号，可以用-b参数
    * -n参数会给所有的行加上行号
    * 不想让制表符出现，可以用-T参数 -T参数会用^I字符组合去替换文中的所有制表符
* less命令的操作和more命令基本一样，一次显示一屏的文件文本。除了支持和more命令相同的命令集，它还包括更多的选项
* tail命令会显示文件最后几行的内容（文件的“尾部”）。默认情况下，它会显示文件的末尾10行
    * 通过加入-n 2使tail命令只显示文件的最后两行 tail -n 2 log_file
    * -f参数是tail命令的一个突出特性。它允许你在其他进程使用该文件时查看文件的内容。tail命令会保持活动状态，并不断显示添加到文件中的内容。这是实时监测系统日志的绝妙方式
* head命令并像tail命令那样支持-f参数特性。 head命令是一种查看文件起始部分内容的便捷方法。
* ps命令只会显示运行在当前控制台下的属于当前用户的进程
    * UID：启动这些进程的用户。
    * PID：进程的进程ID。
    * PPID：父进程的进程号（如果该进程是由另一个进程启动的）。
    * C：进程生命周期中的CPU利用率。
    * STIME：进程启动时的系统时间。
    * TTY：进程启动时的终端设备。
    * TIME：运行进程需要的累计CPU时间。
    * CMD：启动的程序名称
* top命令跟ps命令相似，能够显示进程信息，但它是实时显示的。
* kill命令可通过进程ID（PID）给进程发信号
* killall命令非常强大，它支持通过进程名而不是PID来结束进程。 killall命令也支持通配符，这在系统因负载过大而变得很慢时很有用。
* mount命令提供如下四部分信息：
    * 媒体的设备文件名
    * 媒体挂载到虚拟目录的挂载点
    * 文件系统类型
    * 已挂载媒体的访问状态
* 手动将U盘/dev/sdb1挂载到/media/disk
```bash
mount -t vfat /dev/sdb1 /media/disk
```
* 媒体设备挂载到了虚拟目录后， root用户就有了对该设备的所有访问权限，而其他用户的访问则会被限制。
* umount命令支持通过设备文件或者是挂载点来指定要卸载的设备。如果有任何程序正在使用设备上的文件，系统就不会允许你卸载它
* sort -n file2 告诉sort命令把数字识别成数字而不是字符，并且按值排序
* 如果用-M参数， sort命令就能识别三字符的月份名，并相应地排序
* grep -v t file1 进行反向搜索（输出不匹配该模式的行），可加-v参数
* 如果要显示匹配模式的行所在的行号，可加-n参数
* 如果只要知道有多少行含有匹配的模式，可用-c参数
* 如果要指定多个匹配模式，可用-e参数来指定每个模式
*  gzip：用来压缩文件。
* gzcat：用来查看压缩过的文本文件的内容。
* gunzip：用来解压文件。
* 创建一个归档文件
    * tar -cvf test.tar test/ test2/
* 列出tar文件test.tar的内容
    * tar -tf test.tar
* 从tar文件test.tar中提取内容
    * tar -xvf test.tar
* 文件名以.tgz结尾提取内容
    * tar -zxvf filename.tgz
* 要想将命令置入后台模式，可以在命令末尾加上字符&
* 输入!!，然后按回车键就能够唤出刚刚用过的那条命令来使用
* 命令历史记录被保存在隐藏文件.bash_history中，它位于用户的主目录中。这里要注意的是，bash命令的历史记录是先存放在内存中，当shell退出时才被写入到历史文件中
* 你可以唤回历史列表中任意一条命令。只需输入惊叹号和命令在历史列表中的编号即可 !20
```bash
$ history
[...]
13 pwd
14 ls
15 cd
16 type pwd
17 which pwd
18 type echo
19 which echo
20 type -a pwd
21 type -a echo
[...]
32 history -a
33 history
34 cat .bash_history
35 history
$
$ !20
type -a pwd
pwd is a shell builtin
pwd is /bin/pwd
$
```
* 全局环境变量对于shell会话和所有生成的子shell都是可见的。局部变量则只对创建它们的shell可见。
```bash
$ printenv HOME
/home/Christine
$
$ env HOME
env: HOME: No such file or directory
$
$ echo $HOME
/home/Christine
$ ls $HOME
```

所有的环境变量名均使用大写字母，这是bash shell的标准惯例。如果是你自己创建的局部变量或是shell脚本，请使用小写字母。变量名区分大小写。
在涉及用户定义的局部变量时坚持使用小写字母，这能够避免重新定义系统环境变量可能带来的灾难。

* 变量名、等号和值之间没有空格，这一点非常重要。如果在赋值表达式中加上了空格，bash shell就会把值当成一个单独的命令
```bash
$ my_variable = "Hello World"
-bash: my_variable: command not found
$
```
* 创建全局环境变量的方法是先创建一个局部环境变量，然后再把它导出到全局环境中。这个过程通过export命令来完成，变量名前面不需要加$
```bash
$ my_variable="I am Global now"
$
$ export my_variable
$
$ echo $my_variable
I am Global now
$
$ bash
$
$ echo $my_variable
I am Global now
$
$ exit
exit
$
$ echo $my_variable
I am Global now
$
```
* 既然可以创建新的环境变量，自然也能删除已经存在的环境变量。可以用unset命令完成这个操作。在unset命令中引用环境变量时，记住不要使用$
```bash
$ echo $my_variable
I am Global now
$
$ unset my_variable
$
$ echo $my_variable
```
* bash命令启动了一个子shell
* 修改子shell中全局环境变量并不会影响到父shell中该变量的值
* 子shell甚至无法使用export命令改变父shell中全局环境变量的值
* 如果要用到变量，使用$；如果要操作变量，不使用$。这条规则的一个例外就是使用printenv显示某个变量的值
* 如果你是在子进程中删除了一个全局环境变量，这只对子进程有效。该全局环境变量在父进程中依然可用。
* PATH环境变量定义了用于进行命令和程序查找的目录。
```bash
$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:
/sbin:/bin:/usr/games:/usr/local/games
$
```
* PATH中的目录使用冒号分隔
```bash
$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:
/sbin:/bin:/usr/games:/usr/local/games
$
$ PATH=$PATH:/home/christine/Scripts
$
$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/
games:/usr/local/games:/home/christine/Scripts
$
$ myprog
The factorial of 5 is 120.
$
```
* 如果希望子shell也能找到你的程序的位置，一定要记得把修改后的PATH环境变量导出
* 登录shell会从5个不同的启动文件里读取命令：
    * /etc/profile
    * $HOME/.bash_profile
    * $HOME/.bashrc
    * $HOME/.bash_login
    * $HOME/.profile
* /etc/profile文件是系统上默认的bash shell的主启动文件。系统上的每个用户登录时都会执行这个启动文件
* 最好是在/etc/profile.d目录中创建一个以.sh结尾的文件。把所有新的或修改过的全局环境变量设置放在这个文件中
* 存储个人用户永久性bash shell变量的地方是$HOME/.bashrc文件。这一点适用于所有类型的shell进程
* 可以把自己的alias设置放在$HOME/.bashrc启动文件中，使其效果永久化。
* 环境变量有一个很酷的特性就是，它们可作为数组使用。数组是能够存储多个值的变量。这些值可以单独引用，也可以作为整个数组来引用。
* 要给某个环境变量设置多个值，可以把值放在括号里，值与值之间用空格分隔
* 环境变量数组的索引值都是从零开始
* 要显示整个数组变量，可用星号作为通配符放在索引值的位置
* 用unset命令删除数组中的某个值
* 可以在unset命令后跟上数组名来删除整个数组
* 有时数组变量会让事情很麻烦，所以在shell脚本编程时并不常用
* 在通常的shell脚本中，井号（#）用作注释行。 shell并不会处理shell脚本中的注释行。然而，shell脚本文件的第一行是个例外， #后面的惊叹号会告诉shell用哪个shell来运行脚本
```bash
$ mytest=(one two three four five)
$ $ echo $mytest
one
$ echo ${mytest[2]}
three
$ echo ${mytest[*]}
one two three four five
$ mytest[2]=seven
$
$ echo ${mytest[*]}
one two seven four five
$ unset mytest[2]
$
$ echo ${mytest[*]}
one two four five
$
$ echo ${mytest[2]}
$ echo ${mytest[3]}
four
$
$ unset mytest
$
$ echo ${mytest[*]}
$
```


## Command
name|grammer|function|example
:---:|:---:|:---:|:---:
pwd |pwd|查看当前目录|\
ls|ls -options|显示当前目录下的文件和目录|ls -a, ls -F -R, ls -l, ll
touch|touch name|创建文件|touch test_one
cp|cp source destination|将文件和目录从一个位置复制到另一个位置|\
ln|ln -s data_file sl_data_file|创建符号链接|ln -s data_file sl_data_file|
ln|ln code_file hl_code_file|创建硬链接|ln code_file hl_code_file
mv|mv fall fzll|将文件和目录移动到另一个位置或重新命名|mv fall fzll, mv fzll Pictures/
rm|rm -i fall|删除文件|rm -i fall
mkdir|mkdir -option name|同时创建多个目录和子目录或者单个目录|mkdir -p New_Dir/Sub_Dir/Under_Dir
rmdir|rmdir name|删除目录|rmdir New_Dir
tree|tree name|它能够以一种美观的方式展示目录、子目录及其中的文件|tree Small_Dir
file|file name|探测文件的内部，并决定文件是什么类型的|file my_file
cat|cat test1|显示文本文件中所有数据|cat test1
more|||
less|||
tail|tail log_file|显示文件最后几行的内容|tail log_file
head|head log_file|显示文件开头10行|head log_file
ps -ef||查看系统上运行的所有进程|
top|||
kill|kill 3940|结束进程|kill 3940
killall|killall http*|命令结束了所有以http开头的进程|killall http*
mount|mount -t type device directory|mount命令会输出当前系统上挂载的设备列表|mount -t type device directory
umount|umount [directory | device ]|卸载|umount /home/rich/mnt
df|df -h|某个设备上还有多少磁盘空间|df -h
du||显示某个特定目录（默认情况下是当前目录）的磁盘使用情况|
sort||sort命令是对数据进行排序的|sort file1
grep|grep [options] pattern [file]|搜索数据|grep three file1
gzip|gzip myprog|压缩你在命令行指定的文件|gzip myprog
tar|tar function [options] object1 object2|归档数据|tar function [options] object1 object2
sleep 10|sleep 10|命令sleep 10会将会话暂停10秒钟|sleep 10
jobs||显示后台作业信息|jobs
which ps|||
type -a ps|||
history||可以唤回这些命令并重新使用|
alias|alias -p |查看当前可用的别名|alias li='ls -li'
printenv or env||查看全局变量|env
set||显示为某个特定进程设置的所有环境变量，包括局部变量、全局变量以及用户定义变量|set
useradd|||
userdel|||
usermod|||
passwd|||
chpasswd|||
chsh|||
chfn|||
chage|||
chmod|chmod options mode file|改变文件和目录的安全性设置|chmod 760 newfile
chown|chown options owner[.group] file|改变文件的默认属组|chown dan newfile
chgrp||更改文件或目录的默认属组|chgrp shared newfile
fdisk||管理安装在系统上的任何存储设备上的分区|
date|data|显示了当前日期和时间|date
who||显示当前是谁登录到了系统上|
set||显示一份完整的当前环境变量列表|

