import threading


class Foo:
    def __init__(self):
        self.second_ready = threading.Semaphore(0)
        self.third_ready = threading.Semaphore(0)


    def first(self, printFirst: 'Callable[[], None]') -> None:
        
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.second_ready.release()


    def second(self, printSecond: 'Callable[[], None]') -> None:
        self.second_ready.acquire()
        # printSecond() outputs "second". Do not change or remove this line.
        printSecond()
        self.third_ready.release()


    def third(self, printThird: 'Callable[[], None]') -> None:
        self.third_ready.acquire()
        # printThird() outputs "third". Do not change or remove this line.
        printThird()


foo = Foo()

def printFirst():
    print("first", end='->')

def printSecond():
    print("second", end='->')

def printThird():
    print("third")

# 创建线程
t1 = threading.Thread(target=foo.first, args=(printFirst,))
t2 = threading.Thread(target=foo.second, args=(printSecond,))
t3 = threading.Thread(target=foo.third, args=(printThird,))

# 乱序启动线程，模拟线程无序执行的情况
t3.start()
t1.start()
t2.start()

# 等待所有线程完成
t1.join()
t2.join()
t3.join()