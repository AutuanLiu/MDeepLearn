1. 获得变量值的长度

```bash
length=${#var}
```
2. 检查当前脚本是以超级用户还是以普通用户的身份运行的

```bash
if [ $UID -ne 0 ]; then
    echo Non root user. Please run as root.
else
    echo Root user
fi
```
3. 使用函数添加环境变量
```bash
prepend() { [ -d "$2" ] && eval $1=\"$2\$\{$1:+':'\$$1\}\" && export $1 ; }
# prepend PATH /opt/myapp/bin
```
4. 数学运算
* 在Bash shell环境中，可以利用let、 (( ))和[]执行基本的算术操作。而在进行高级操作时，expr和bc这两个工具也会非常有用
* let命令可以直接执行基本的算术操作。当使用let时，变量名之前不需要再添加$

```bash
let no1++
let no+=6
result=$[ no1 + no2 ]
result=$(( no1 + 50 ))
result=`expr 3 + 4`
result=$(expr $no1 + 5)
result=`echo "$no * 1.5" | bc`
echo "scale=2;3/8" | bc
# 进制转换
"obase=2;$no" | bc
```
5. 文件描述符

* 文件描述符是与某个打开的文件或数据流相关联的整数。文件描述符0、 1以及2是系统预留的。
    * 0 —— stdin（标准输入）。
    * 1 —— stdout（标准输出）。
    * 2 —— stderr（标准错误）

6. 数组
```bash
echo ${array_var[*]}
echo ${array_var[@]}
# 打印数组长度
echo ${#array_var[*]}
```
7. 关联数组
```bash
fruits_value=([apple]='100dollars' [orange]='150 dollars')
```
```bash
# 在两个文件中复制内容
tee file1 file2
```
8. 列出数组索引
```bash
echo ${!array_var[*]}
echo ${!array_var[@]
```
9. 使用set -x和set +x对脚本进行部分调试
```bash
for i in {1..6}; do
    set -x
    echo $i
    set +x
done
```
* set –x：在执行时显示参数和命令。
* set +x：禁止调试。
* set –v：当命令进行读取时显示输入。
* set +v：禁止打印输入
* 从#!/bin/bash改成 #!/bin/bash -xv，这样一来，不用任何其他选项就可以启用调试功能了

```bash
fname(){
    echo $1, $2; #访问参数1和参数2
    echo "$@";   #以列表的方式一次性打印所有参数
    echo "$*";   #类似于$@，但是参数被作为单个实体
    return 0;    #返回值
}
```
* "$@" 被扩展成 "$1" "$2" "$3"等。
* "$*" 被扩展成 "$1c$2c$3"，其中c是IFS的第一个字符。
* "$@" 要比"$*"用得多。由于 "$*"将所有的参数当做单个字符串，因此它很少被使用
10. 执行程序直到成功
```bash
repeat() { while :; do $@ && return; done }
# 命令每30秒运行一次
repeat() { while :; do $@ && return; sleep 30; done }
```
11. find
* find命令的工作方式如下：沿着文件层次结构向下遍历，匹配符合条件的文件，执行相应的操作
```bash
find /home/slynux -name "*.txt" -print
```
12. 结合find使用xargs
```bash
find . -type f -name "*.txt" -print | xargs rm -f
# 统计源代码目录中所有C程序文件的行数
find source_code_dir_path -type f -name "*.c" -print0 | xargs -0 wc -l
```
13. tr
* tr只能通过stdin（标准输入），而无法通过命令行参数来接受输入
* 输入字符由大写转换成小写
```bash
echo "HELLO WHO IS THIS" | tr 'A-Z' 'a-z'
```
14. 校验和与核实
```bash
# 计算md5sum
md5sum filename
# 这个命令会输出校验和是否匹配的消息
md5sum -c file_sum.md5
```
15. 用gpg加密文件
```bash
gpg -c filename
```
16. uniq
* uniq命令通过消除重复内容，从给定输入中（stdin或命令行参数文件）找出唯一的行。它也可以用来找出输入中出现的重复行
```bash
uniq sorted.txt
# 只显示唯一的行（在输入文件中没有重复出现的行）：
$ uniq -u sorted.txt
sort unsorted.txt | uniq -u
```
17. 分割文件和数据
* 将该文件分割成多个大小为10KB的文件
```bash
split -b 10k data.file
split -b 10k data.file -d -a 4
```
18. 根据扩展名切分文件名
```bash
file_jpg="sample.jpg"
name=${file_jpg%.*}
echo File name is: $name
```
19. 将文件名的扩展名部分提取出
```bash
extension=${file_jpg#*.}
echo Extension is: jpg
```
20. 批量重命名
```bash
#!/bin/bash
#文件名: rename.sh
#用途: 重命名 .jpg 和 .png 文件
count=1;
for img in `find . -iname '*.png' -o -iname '*.jpg' -type f -maxdepth 1`
do
new=image-$count.${img##*.}
echo "Renaming $img to $new"
mv "$img" "$new"
let count++
done
```
21. 批量生成空白文件
```bash
for name in {1..100}.txt; do
    touch $name
done
```
22. 用 cut 按列切分文件
```bash
cut -f FIELD_LIST filename
cut -f 2,3 filename
```
23. 使用 sed 进行文本替换
```bash
sed 's/pattern/replace_string/' file
cat file | sed 's/pattern/replace_string/'
# 在替换的同时保存更改
sed -i 's/text/replace/' file
```
24. awk
```bash
# 打印每一行的第2和第3个字段
awk '{ print $3,$2 }' file
# 统计文件中的行数
awk 'END{ print NR }' file
# 将每一行中第一个字段的值进行累加
seq 5 | awk 'BEGIN{ sum=0; print "Summation:" }
{ print $1"+"; sum+=$1 } END { print "=="; print sum }'
```
25. 统计特定文件中的词频
```bash
#!/bin/bash
# 文件名： word_freq.sh
# 用途: 计算文件中单词的词频
if [ $# -ne 1 ]; then
    echo "Usage: $0 filename";
    exit -1
fi
filename=$1
egrep -o "\b[[:alpha:]]+\b" $filename | awk '{ count[$0]++ }
END{ printf("%-14s%s\n","Word","Count") ;
for(ind in count)
{ printf("%-14s%d\n",ind,count[ind]); }
}'
```
26. 按列合并多个文件
```bash
# -d明确指定定界符
paste file1.txt file2.txt -d ","
```
27. 打印文件或行中的第 n 个单词或列
```bash
awk '{ print $5 }' filename
```
28. 打印当前目录下各文件的权限和文件名
```bash
ls -l | awk '{ print $1 " : " $8 }'
```
29. 打印行或样式之间的文本
```bash
# 打印出从M行到N行这个范围内的所有文本
awk 'NR==M, NR==N' filename
cat filename | awk 'NR==M, NR==N'
seq 100 | awk 'NR==4,NR==6'
```
30. 以逆序形式打印行
```bash
seq 5 | tac
# 使用awk的实现方式
seq 9 | \
awk '{ lifo[NR]=$0 }
END{ for(lno=NR;lno>-1;lno--){ print lifo[lno]; }
}'
```
31. 在文件中移除包含某个单词的句子
```bash
sed 's/ [^.]*mobile phones[^.]*\.//g' sentence.txt
```
32. 文本切片及参数操作
```bash
# 打印第5个字符之后的内容：
string=abcdefghijklmnopqrstuvwxyz
echo ${string:4}
# 从第5个字符开始，打印8个字符：
echo ${string:4:8}
echo ${string:(-1)}
echo ${string:(-2):2}
```
33.  拼接两个归档文件
```bash
tar -Af file1.tar file2.tar
```
34. 比较归档文件与文件系统中的内容
```bash
tar -df archive.tar
```
35. 从归档文件中删除文件
```bash
tar -f archive.tar --delete file1 file2 ..
tar --delete --file archive.tar [FILE LIST]
```
36. 创建SSH密钥
```bash
ssh-keygen -t rsa
```
37. 添加一个密钥文件
```bash
ssh USER@REMOTE_HOST "cat >> ~/.ssh/authorized_keys" < ~/.ssh/id_rsa.pub
```
38. 自动将私钥加入远
程服务器的authorized_keys文件中
```bash
ssh-copy-id USER@REMOTE_HOST
```
39. 将位于远程主机上的文件系统挂载到本地挂载点上
```bash
sshfs -o allow_other user@remotehost:/home/path /mnt/mountpoint
```
40. 图像文件的缩放及格式转换
```bash
# 将一种图像格式转换为另一种图像格式
convert INPUT_FILE OUTPUT_FILE
# 指定WIDTH（宽度）或HEIGHT（高度）来缩放图像
convert image.png -resize WIDTHxHEIGHT image.png
```
