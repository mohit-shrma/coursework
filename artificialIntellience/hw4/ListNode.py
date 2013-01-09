#node contains test condition to split and maximum upto
#two attributes involved
class ListNode:

    def __init__(self, testAttributes = {}, isIncl=True,\
                     connector = '', predictedLabel = None):
        
        #a dict containing attributes along with value for the test
        self.testAttributes = testAttributes
        #a dict indicating whether to include/exclude elements satisfying
        #attributes with vals, k => v where v in {'', '~'}
        #'~' -> indicates exclude others ,
        #'' -> include only those satisfying attribute val
        self.isIncl = isIncl
        #connector operator i.e 'or' 'and', in case of two attributes
        self.connector = connector 
        #predicted label if test satisfied
        self.predictedLabel = predictedLabel
        #next list node in case not satisfied
        self.next = None

    def setNextNode(self, nextNode):
        self.next = nextNode

        
