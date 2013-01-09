#node contains test attribute to split, branches and
#predicted label in case of leaf nodes
class TreeNode:

    def __init__(self, testAttribute=None, predictedLabel=None):
        #at a time either first or both of them is passed
        self.testAttribute = testAttribute
        self.predictedLabel = predictedLabel
        #branches contains label, subtree pair 
        self.branches = {}

    def addSubtree(self, attribVal, treeNode):
        self.branches[attribVal] = treeNode

