"""
   CSci5512 Spring'12 Homework 4
   login: sharm163@umn.edu
   date: 4/30/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: dtree4
"""

from Example import Example, Attribute
from TreeNode import TreeNode
import math
import sys

#check if all examples have same class
def checkIfSameClass(examples):
    firstLabel = examples[0].label
    for i in range(len(examples)):
        if firstLabel != examples[i].label:
            return False
    return True

#returns majority labels from examples
def majorityValue(examples):
    labelDict = {}
    for i in range(len(examples)):
        if examples[i].label in labelDict:
            labelDict[examples[i].label] += 1
        else:
            labelDict[examples[i].label] = 0
    maxKey = ''
    maxVal = -1
    for key, val in labelDict.iteritems():
        if val > maxVal:
            maxKey = key
            maxVal = val
    return maxKey

def getInfoConent(listVal):
    sum = 0
    listVal = [float(val) for val in listVal]
    for val in listVal:
        if val != 0:
            sum += (-1*val)*math.log(val, 2)  
    return sum

#
def getEntropy(attribute, examples):
    entropy = 0
    #TODO:can be optimized further
    for attribVal in Attribute.getAttributes()[attribute]:
        posEx = 0
        negEx = 0
        for example in examples:
            if example.attributes[attribute] == attribVal:
                if example.label == True:
                    posEx += 1
                else:
                    negEx += 1
        totalAttribEx = float(posEx + negEx)
        if totalAttribEx != 0:
            aInfo, bInfo = float(posEx)/totalAttribEx, float(negEx)/totalAttribEx
            entropy += ( totalAttribEx/len(examples) )\
                *getInfoConent([aInfo, bInfo])
    return entropy
        
    

#find the best attribute of given attributes to split on
def chooseAttribute(attributes, examples):
    #identify the attributes to split examples based on entropy
    minEntropyVal = 100
    minEntropyAttribute = ''
    posExCount = 0
    for example in examples:
        if example.label == True:
            posExCount += 1
    negExCount = len(examples) - posExCount
    for attribute in attributes:
        tempEntropy = getEntropy(attribute, examples)
        if tempEntropy <= minEntropyVal:
            minEntropyVal = tempEntropy
            minEntropyAttribute = attribute
    return minEntropyAttribute
    

#find examples with given value for a attribute
def filterExamples(attribute, value, examples):
    filteredExamples = []
    for example in examples:
        if example.attributes[attribute] == value:
            filteredExamples.append(example)
    return filteredExamples

def filterAttributes(attributes, filterAttribute):
    found = -1
    for i in range(len(attributes)):
        if attributes[i] == filterAttribute:
            found = i
            break
    if found != -1:
        del attributes[found]
    return attributes

#return a decision tree, for examples, attributes val
#and default vaue for goal predicate
def decisionTreeLearning(examples, attribs, default):
    if len(examples) == 0:
        #if example list empty then return the majority value
        #computed previously
        return TreeNode(None, default)
    elif checkIfSameClass(examples):
        #if all examples same class then return the class
        return TreeNode(None, examples[0].label)
    elif len(attribs) == 0:
        #if attributes empty then return majority value
        return TreeNode(None, majorityValue(examples))
    else:
        bestAttrib = chooseAttribute(attribs, examples)
        tree = TreeNode(bestAttrib)
        majorityVal = majorityValue(examples)
        #for each possible value of attribute
        for attribVal in Attribute.getAttributes()[bestAttrib]:
            filteredExamples = filterExamples(bestAttrib, attribVal, examples)
            subTree = decisionTreeLearning(filteredExamples, \
                         filterAttributes(attribs, bestAttrib), majorityVal)
            tree.addSubtree(attribVal, subTree)
        return tree
        
#prints learned tree by applying BFS
def printDecisionTree(rootNode):
    queue = []
    queue.append((rootNode, 'root'))
    while len(queue) != 0:
        node , parentLabel = queue.pop()
        if node.testAttribute:
            print 'test: ', 'path from root: [', parentLabel, \
                '] new test attribute:', node.testAttribute
            if len(node.branches):
                strBranches = '[' + parentLabel + ']: new branches: ['
                for branchLabel, subTree in node.branches.iteritems():
                    strBranches +=  str(branchLabel) + ','
                    queue.insert(0, (subTree, \
                                        parentLabel + ':' +str(branchLabel)))
                print strBranches+']'
        elif node.predictedLabel is not None:
            print 'prediction: path from root [', parentLabel+']'\
                , 'predicted label:', node.predictedLabel

def predictFromDecisionTree(node, testAttribs):
    if node.testAttribute:
        toFollowBranch = testAttribs[node.testAttribute]
        return predictFromDecisionTree(node.branches[toFollowBranch], testAttribs)
    else:
        return node.predictedLabel

def getAccuracy(examples, rootNode):
    predictions = [predictFromDecisionTree(rootNode, example.attributes)\
                       for example in examples]
    equalCount = 0
    for i in range(len(predictions)):
        if examples[i].label ==  predictions[i]:
            equalCount += 1
    return 100 * (float(equalCount))/len(examples)

    
def main():
    examples = []
    examples.append(Example({'alt':True, 'bar':False, 'fri':False, 'hun':True,\
                                 'pat':'some', 'price':'$$$', 'rain':False,\
                                 'res':True, 'type':'french', 'est':'0-10'},\
                                True))
    examples.append(Example({'alt':True, 'bar':False, 'fri':False, 'hun':True,\
                                 'pat':'full', 'price':'$', 'rain':False,\
                                 'res':False, 'type':'thai', 'est':'30-60'},\
                                False))
    examples.append(Example({'alt':False, 'bar':True, 'fri':False, 'hun':False,\
                                 'pat':'some', 'price':'$', 'rain':False,\
                                 'res':False, 'type':'burger', 'est':'0-10'},\
                                True))
    examples.append(Example({'alt':True, 'bar':False, 'fri':True, 'hun':True,\
                                 'pat':'full', 'price':'$', 'rain':True,\
                                 'res':False, 'type':'thai', 'est':'10-30'},\
                                True))
    examples.append(Example({'alt':True, 'bar':False, 'fri':True, 'hun':False,\
                                 'pat':'full', 'price':'$$$', 'rain':False,\
                                 'res':True, 'type':'french', 'est':'>60'},\
                                False))
    examples.append(Example({'alt':False, 'bar':True, 'fri':False, 'hun':True,\
                                 'pat':'some', 'price':'$$', 'rain':True,\
                                 'res':True, 'type':'italian', 'est':'0-10'},\
                                True))
    examples.append(Example({'alt':False, 'bar':True, 'fri':False, 'hun':False,\
                                 'pat':'none', 'price':'$', 'rain':True,\
                                 'res':False, 'type':'burger', 'est':'0-10'},\
                                False))
    examples.append(Example({'alt':False, 'bar':False, 'fri':False, 'hun':True,\
                                 'pat':'some', 'price':'$$', 'rain':True,\
                                 'res':True, 'type':'thai', 'est':'0-10'},\
                                True))
    examples.append(Example({'alt':False, 'bar':True, 'fri':True, 'hun':False,\
                                 'pat':'full', 'price':'$', 'rain':True,\
                                 'res':False, 'type':'burger', 'est':'>60'},\
                                False))
    examples.append(Example({'alt':True, 'bar':True, 'fri':True, 'hun':True,\
                                 'pat':'full', 'price':'$$$', 'rain':False,\
                                 'res':True, 'type':'italian', 'est':'10-30'},\
                                False))
    examples.append(Example({'alt':False, 'bar':False, 'fri':False, 'hun':False,\
                                 'pat':'none', 'price':'$', 'rain':False,\
                                 'res':False, 'type':'thai', 'est':'0-10'},\
                                False))
    examples.append(Example({'alt':True, 'bar':True, 'fri':True, 'hun':True,\
                                 'pat':'full', 'price':'$', 'rain':False,\
                                 'res':False, 'type':'burger', 'est':'30-60'},\
                                True))
    rootNode = decisionTreeLearning(examples, Attribute.getAttributes().keys(),\
                                        False)
    printDecisionTree(rootNode)

    print predictFromDecisionTree(rootNode, {'alt':True, 'bar':True, 'fri':True, 'hun':True,\
                                 'pat':'full', 'price':'$', 'rain':False,\
                                 'res':False, 'type':'burger', 'est':'30-60'})
    
    print '**** prediction accuracy ****'
    trainAccuracy = getAccuracy(examples, rootNode) 
    print 'training set error: ' + str(100-trainAccuracy) + '%'


if __name__ == '__main__':
    main()
