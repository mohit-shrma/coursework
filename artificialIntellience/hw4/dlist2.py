"""
   CSci5512 Spring'12 Homework 4
   login: sharm163@umn.edu
   date: 4/30/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: dlist2
"""

import sys
from Example import Example, Attribute
from ListNode import ListNode

class DListCreationErr(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


#check if gives pure examples and how many (bool, count, predLabel) for the given attribute value
# and also to include that pair or exclude that
#plz ignore bInclude parameter this was added to do negation
# its not currently being used anywhere
def ifPureExamples(examples, attribute, val, bInclude):

    #list to store labels of satisfying examples
    labels = []
    
    #list to store labels of satisfying examples
    notLabels = []
    
    for example in examples:
        if example.attributes[attribute] == val:
            labels.append(example.label)
        else:
            notLabels.append(example.label)
    if bInclude:
        if len(set(labels)) == 1:
            return (True, len(labels), labels[0])
        else:
            return (False, len(labels), '')
    else:
        if len(set(notLabels)) == 1:
            return (True, len(notLabels), notLabels[0])
        else:
            return (False, len(notLabels), '')


#accept a key value pair for two attributes in attributesDict
#plz ignore bInclude parameter this was added to do negation
# its not currently being used anywhere
def ifPureExamples2(examples, attributesDict, bInclude):

    #list to store labels of satisfying examples
    labels = []
    
    #list to store labels of satisfying examples
    notLabels = []

    attribute1 = attributesDict.keys()[0]
    val1 = attributesDict[attribute1]
    
    attribute2 = attributesDict.keys()[1]
    val2 = attributesDict[attribute2]
    
    #first try with 'and' '^' operator
    for example in examples:
        if example.attributes[attribute1] == val1 and\
                example.attributes[attribute2] == val2:
            labels.append(example.label)
        else:
            notLabels.append(example.label)
    
    if bInclude:
        if len(set(labels)) == 1:
            return (True, len(labels), labels[0], 'and')
        else:
            return (False, len(labels), '', '')
    else:
        if len(set(notLabels)) == 1:
            return (True, len(notLabels), notLabels[0], 'and')
        else:
            return (False, len(notLabels), '', '')


#find a suitable tests out of passed examples
def findTest(examples, attributesDict):

    selectedAttributes = {}
    selectedPureCount = 0
    isSelectedIncl = True
    selectedPureLabel = ''
    selectedOp = ''
    
    #first look for test of 1 attribute only
    for attribute, vals in attributesDict.iteritems():
        for value in vals:

            isPure, count, label = ifPureExamples(examples, attribute, value, True)
            if isPure and count > selectedPureCount:
                selectedPureCount = count
                selectedAttributes = {attribute:value}
                isSelectedIncl = True
                selectedPureLabel = label

    #try it for other combination of attributes
    if True:#len(selectedAttributes) == 0:
        #attribute not found, try combination of attributes i.e two at a time
        attributeKeys = attributesDict.keys()

        for i in range(len(attributeKeys)):
            attrib1 = attributeKeys[i]
            for j in range(i+1, len(attributeKeys)):
                attrib2 = attributeKeys[j]
                for val1 in attributesDict[attrib1]:
                    for val2 in attributesDict[attrib2]:
                        attribDict = {attrib1:val1, attrib2:val2}
                        isPure, count, label, op = ifPureExamples2(examples,\
                                                                       attribDict, True)
                        if isPure and count > selectedPureCount:
                            selectedPureCount = count
                            selectedAttributes = {attrib1:val1, attrib2:val2}
                            isSelectedIncl = True
                            selectedPureLabel = label
                            selectedOp = op
                    
    return (selectedAttributes, isSelectedIncl, selectedPureLabel, selectedOp)


def filterExamples(examples, selectedAttributes, isSelectedIncl, selectedOp):
    filteredExamples = []
    if len(selectedAttributes) == 1:
        #only one attribute satisfy the test
        attrib = selectedAttributes.keys()[0]
        val = selectedAttributes[attrib]
        for example in examples:
            if isSelectedIncl:
                if example.attributes[attrib] != val:
                    filteredExamples.append(example)
            else:
                if example.attributes[attrib] == val:
                    filteredExamples.append(example)
    elif len(selectedAttributes) == 2:
        #two attributes chosen for test
        attrib1 = selectedAttributes.keys()[0]
        val1 = selectedAttributes[attrib1]
        attrib2 = selectedAttributes.keys()[1]
        val2 = selectedAttributes[attrib2]
        for example in examples:
            if selectedOp == 'and':
                if isSelectedIncl:
                    if not (example.attributes[attrib1] == val1 and\
                        example.attributes[attrib2] == val2):
                        filteredExamples.append(example)
                else:
                    if (example.attributes[attrib1] == val1 and\
                        example.attributes[attrib2] == val2):
                        filteredExamples.append(example)
    return filteredExamples

def decisionListLearning(examples):
    #if examples are empty then return trivial list node
    #with predicted label as 'No'/False
    if len(examples) == 0:
        return ListNode({}, {}, {}, False)
    #find a test S.T. subset of examples either all +ve or -ve
    (selectedAttributes, isSelectedIncl, selectedPureLabel, selectedOp) \
        = findTest(examples, Attribute.getAttributes())
    if not len(selectedAttributes):
        raise(DListCreationErr('couldn\'t find attributes for even split.'))

    listNode =  ListNode(selectedAttributes, isSelectedIncl, selectedOp,\
                             selectedPureLabel)
    #set next node from list learned by remaining examples
    listNode.setNextNode(decisionListLearning(filterExamples(examples,\
                                                            selectedAttributes,\
                                                            isSelectedIncl,\
                                                            selectedOp)))
    return listNode

def printDecisionList(node):
    print '<------------------ Node ---------------------->'
    if len(node.testAttributes):
        for attribute, val in node.testAttributes.iteritems():
            print 'attribute: '+ str(attribute) + ' value: ' + str(val)
    else:
        print 'End'
    if node.predictedLabel is not None:
        print 'label:', node.predictedLabel
    if node.next is not None:
        printDecisionList(node.next)
        
def predictFromDecisionList(example, node):
    if len(node.testAttributes) == 1:
        attrib1 = node.testAttributes.keys()[0]
        val1 = node.testAttributes[attrib1]
        if example.attributes[attrib1] == val1:
            if node.isIncl:
                return node.predictedLabel
            else:
                #go to next list in node
                return predictFromDecisionList(example, node.next)
        else:
            if node.isIncl:
                #go to next list in node
                return predictFromDecisionList(example, node.next)
            else:
                return node.predictedLabel
    elif len(node.testAttributes) == 2:
        
        attrib1 =  node.testAttributes.keys()[0]
        val1 = node.testAttributes[attrib1]
        exVal1 = example.attributes[attrib1]
        
        attrib2 = node.testAttributes.keys()[1]
        val2 = node.testAttributes[attrib2]
        exVal2 = example.attributes[attrib2]
        
        nodeOp = node.connector

        if nodeOp == 'and':
            #operation is and
            if val1 == exVal1 and val2 == exVal2:
                if node.isIncl:
                    return node.predictedLabel
                else:
                    return predictFromDecisionList(example, node.next)
            else:
                if node.isIncl:
                    return predictFromDecisionList(example, node.next)
                else:
                    return node.predictedLabel
    else:
        #termination node
        return node.predictedLabel


def getAccuracy(examples, rootNode):
    predictions = [predictFromDecisionList(example, rootNode)\
                       for example in examples]
    equalCount = 0
    for i in range(len(predictions)):
        if examples[i].label ==  predictions[i]:
            equalCount += 1
    return 100*(float(equalCount))/len(examples)
    
    
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
    rootNode = decisionListLearning(examples)
    print '***** decision list *****'
    printDecisionList(rootNode)
    print '***** prediction accuracy ******'
    trainAccuracy = getAccuracy(examples, rootNode) 
    print 'training set error: ' + str(100-trainAccuracy) + '%'

if __name__ == '__main__':
    main()

        
