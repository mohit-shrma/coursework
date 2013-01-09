
#define a class for containing attributes, equivalent to ENUMS
class Attribute:

    #TODO: in comments what type of value or how many values can these accept
    AttributesDict = {'alt':[True, False], 'bar':[True, False],\
                          'fri':[True, False], 'hun':[True, False],\
                          'pat':['some', 'full', 'none'],\
                          'price':['$', '$$', '$$$'],\
                          'rain':[True, False], 'res':[True, False],\
                          'type':['french','thai','burger', 'italian'],\
                          'est':['0-10', '30-60', '10-30', '>60']}
    @staticmethod
    def getAttributes():
        return Attribute.AttributesDict

#TODO: above attribute info store in class Example only
class Example:

    #takes attributes value dict as argument, where is attribute value is one of
    #above 
    def __init__(self, attributesValDict, label):
        self.attributes = {}
        attributeDict =  Attribute.getAttributes()
        attributeKeys = attributeDict.keys()
        if len(attributeKeys) != len(attributesValDict):
            #TODO:raise some exception, can't be initialized correctly
            print 'err'
        else:
            for key, val in attributesValDict.iteritems():
                if key in attributeKeys and (val in attributeDict[key]):
                    self.attributes[key] = val
                else:
                    #raise error
                    print 'err', key, val
        self.label = label
        
