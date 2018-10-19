

class Student(object):

    def __init__(self):
        pass
        # self.name = name

    def __call__(self,dd):
        print('My name is %s.' % dd)

#
# s = Student('dd')
# print(callable(s))
# s('dasd')
# #
# # s('duanyuefeng')
# print(callable(s))

for i in range(len(dir(Student))):
    if dir(Student)[i] not in dir(object):
        print(dir(Student)[i])
print(len(dir(Student)))
print(len(dir(object)))