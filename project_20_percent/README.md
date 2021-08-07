------------20% mark Project (Computer Vision)-------------------

1. Purpose:

2. User manual:
    a. Create data: go to this file directory and type command line:
        >> python CreateData.py <capture-method> <image-height> <image-width>
        Example: >> python CreateData.py screen 500 500
        Note: <capture-method>: cam for web cam image capture (make sure to enable your web cam)
                                screen for screen image capture
    b. Delete error data: go to this file directory and type command line:
        >> python ClearErrorData.py <first-num> <last-num>
        Example: >> python ClearErrorData.py 300 500
        Note: <first-num> first count number of data you want to delete
              <last-num> last count number of data you want to delete 
              (empty if you want to delete from <first-num> to the end of count number)
