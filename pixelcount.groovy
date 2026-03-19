// Get all annotations in the image
def annotations = getAnnotationObjects()

// Optional: If you only want to count a specific class, remove the "//" from the next line and put your class name
// annotations = annotations.findAll{it.getPathClass() == getPathClass("Your_Class_Name")}

double totalPixels = 0

// Loop through them and add up the raw pixel area
annotations.each {
    if (it.getROI() != null) {
        totalPixels += it.getROI().getArea()
    }
}

print "Total annotation pixel count: " + totalPixels
