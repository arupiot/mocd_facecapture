/*
 *  Ministry of Common Data Face Capture
 */

/* booleans triggers */
var showCLMPoints = false;
var showImage = true;
var showFaceMesh = true;
var showBoundingBox = false;
var showFullscreen = false;
var showText = false;
var showFeatures = true;

/* media */
var myCamera;
var mv;
var aspectX, aspectY;
var startSound;
var fontAstronaut;

/* face detection */
var facemesh;
var predictions = [];
let ready = false;
var label = 'no recorded faces';

/* knn classification */
var knn;
var features;

/* preload assets */
function preload() {
//   soundFormats('mp3', 'ogg');
//   startSound = loadSound('assets/audio/bw_mocd_avatar_tribal_drum_start');
//   startSound.playMode('untilDone');
  fontAstronaut = loadFont('assets/font/astronaut.ttf');
}

/* notify when the FaceMesh model has been loaded and is ready */
function modelReady() {
    console.log("FaceMesh model ready");
    console.log('number of labels: '+knn.getNumLabels());
}

function goClassify() {
    const logits = features.infer(myCamera);
    knn.classify(logits, function(error, result) {
      if (error) {
        console.error(error);
      } else {
        label = result.label;
        console.log("identified face: "+label);
        // ready = false;
        goClassify();
      }
    });
    // ready = false;
  }

function learnFace(faceCode) {
    const logits = features.infer(myCamera);
    knn.addExample(logits, faceCode);
    console.log('learning face with code: '+faceCode);
}

// function uuidv4() {
//     return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
//         (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
//     );
// }

// function uuid() {
//     return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
//       var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
//       return v.toString(16);
//     });
//   }

function uuid8() {
    return 'xxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

function setup() {

    /* load HTML canvas */
    //createCanvas(windowWidth, windowHeight);
    loadCanvas(windowWidth, windowHeight);
    
    /* webcam */
    // myCamera = loadCamera(windowWidth, windowHeight, false);
    myCamera = loadCameraWH(VIDEO, windowWidth, windowHeight, true);
    // myCamera = loadCameraWH("http://127.0.0.1:5002/cam.mjpg", windowWidth, windowHeight, true);
    // myCamera = createImg("http://127.0.0.1:5002/cam.mjpg");
    // myCamera.hide();
    
    // var dc = document.getElementById("defaultCanvas0");
    // console.log(dc.width+"-"+dc.height)

    var mv = document.getElementById("v");
    console.log(mv.width+"-"+mv.height)
    
    aspectX = windowWidth / mv.width;
    aspectY = windowHeight / mv.height;
    
    // aspectX = windowWidth / myCamera.width;
    // aspectY = windowHeight / myCamera.height;
    console.log(aspectX, aspectY);

    /* load cml face tracker */
    loadTracker();

    /* load FaceMesh tracker */
    facemesh = ml5.facemesh(myCamera, modelReady);
    /* set up an event that fills the global variable "predictions" with an array every time new predictions are made */
    facemesh.on("predict", results => {
      predictions = results;
    });

    /* KNN classifier */
    features = ml5.featureExtractor('MobileNet', modelReady);
    knn = ml5.KNNClassifier();

    /* load font */
    textFont(fontAstronaut);
    textSize(windowHeight / 20);
    textAlign(CENTER, CENTER);
}
      
function draw() {
    // aspectX = windowWidth / myCamera.width;
    // aspectY = windowHeight / myCamera.height;

    /* clear the screen */
    clear();
    
    var mv = document.getElementById("v");
    aspectX = windowWidth / mv.width;
    aspectY = windowHeight / mv.height;

    /* show background image if chosen */
    // if (showImage == true) image(myCamera,windowWidth/2,windowHeight/2,windowWidth,windowHeight); 
    if (showImage == true) {
        // var mv = document.getElementById("v");
        // aspectX = windowWidth / mv.width;
        // aspectY = windowHeight / mv.height;
        // image(myCamera,0,0,windowWidth*aspectX,windowHeight*aspectY); 
        image(myCamera,0,0,windowWidth,windowHeight); 
        filter(GRAY);
    }
    // if (showImage == true) myCamera.show();
    // if (showImage == false) myCamera.hide();

    /* get CLM face markers */
    getPositions();

    /* is there a detected face? */
    if (predictions.length > 0) {
        /* there is a detected face */

        /* is there an already learned face? */
        // if (!ready && knn.getNumLabels() > 0) {
        if (knn.getNumLabels() > 0) {
            /* there is a face already learned */
            /* look up for learned faces */
            goClassify();
            /* print face code */
            text(label, windowWidth/2, windowHeight/2);  
            if (showCLMPoints == true) drawCLMPoints();
            if (showFaceMesh == true) drawFaceMeshKeypoints();
            if (showBoundingBox == true) drawFaceMeshBoundingBox();
            if (showFeatures == true) drawFaceMeshFeatures();
            // ready = true;
        } else {
            /* there isn't a face already learned */
            /* generate random face code */
            const faceCode = uuid8();
            text("learning face: "+faceCode, windowWidth/2, windowHeight/2);  
            /* learn face */
            for (let i = 0; i < 10; i+=1) {
                learnFace(faceCode);
                // save(knn, 'model.json');
            }
        }
    } else {
        /* there is no detected face */
        /* show no face detected text */
        fill(255,255,255);
        noStroke();
        /* show no face detected text */
        text('no face detected', windowWidth/2, windowHeight/2);  
    }
    

    /* print instructions */
    if (showText ==true) {
        fill(255,255,255);
        noStroke();
        text('"i": image / "p": points / "m" mesh / "b": box / "f": features / "s": fullscreen', windowWidth/2, windowHeight-windowHeight/20*2);  
        // text(aspectX+'-'+aspectY, windowWidth/2, windowHeight-windowHeight/20);  
    }
        
    // console.log(myCamera.width+"-"+myCamera.height+"-"+windowWidth+"-"+windowHeight);
}

function drawFaceMeshFeatures() {
    for (let i = 0; i < predictions.length; i += 1) {
        const annotations = predictions[i].annotations;
        
        const silhouette = annotations.silhouette; // polygon
        drawFeatureLine(silhouette, true);

        const lipsLowerInner = annotations.lipsLowerInner; // polygon
        drawFeatureLine(lipsLowerInner, true);

        const lipsLowerOuter = annotations.lipsLowerOuter; // polygon
        drawFeatureLine(lipsLowerOuter, false);

        const lipsUpperInner = annotations.lipsUpperInner; // polygon
        drawFeatureLine(lipsUpperInner, false); 

        const lipsUpperOuter = annotations.lipsUpperOuter; // polygon
        drawFeatureLine(lipsUpperOuter, false);

        const leftEyeLower0 = annotations.leftEyeLower0; // polygon
        drawFeatureLine(leftEyeLower0, false);
        
        const leftEyeLower1 = annotations.leftEyeLower1; // polygon
        drawFeatureLine(leftEyeLower1, false);

        const leftEyeLower2 = annotations.leftEyeLower2; // polygon
        drawFeatureLine(leftEyeLower2, false);

        const leftEyeLower3 = annotations.leftEyeLower3; // polygon
        drawFeatureLine(leftEyeLower3, false);

        const leftEyeUpper0 = annotations.leftEyeUpper0; // polygon
        drawFeatureLine(leftEyeUpper0, false);

        const leftEyeUpper1 = annotations.leftEyeUpper1; // polygon
        drawFeatureLine(leftEyeUpper1, false);

        const leftEyeUpper2 = annotations.leftEyeUpper2; // polygon
        drawFeatureLine(leftEyeUpper2, false);

        const rightEyeLower0 = annotations.rightEyeLower0; // polygon
        drawFeatureLine(rightEyeLower0, false);

        const rightEyeLower1 = annotations.rightEyeLower1; // polygon
        drawFeatureLine(rightEyeLower1, false);

        const rightEyeLower2 = annotations.rightEyeLower2; // polygon
        drawFeatureLine(rightEyeLower2, false);

        const rightEyeLower3 = annotations.rightEyeLower3; // polygon
        drawFeatureLine(rightEyeLower3, false);

        const rightEyeUpper0 = annotations.rightEyeUpper0; // polygon
        drawFeatureLine(rightEyeUpper0, false);

        const rightEyeUpper1 = annotations.rightEyeUpper1; // polygon
        drawFeatureLine(rightEyeUpper1, false);

        const rightEyeUpper2 = annotations.rightEyeUpper2; // polygon
        drawFeatureLine(rightEyeUpper2, false);

        const leftEyebrowLower = annotations.leftEyebrowLower; // polygon
        drawFeatureLine(leftEyebrowLower, false);

        const leftEyebrowUpper = annotations.leftEyebrowUpper; // polygon
        drawFeatureLine(leftEyebrowUpper, false);

        const rightEyebrowLower = annotations.rightEyebrowLower; // polygon
        drawFeatureLine(rightEyebrowLower, false);

        const rightEyebrowUpper = annotations.rightEyebrowUpper; // polygon
        drawFeatureLine(rightEyebrowUpper, false);

        const noseBottom = annotations.noseBottom; // point
        const noseTip = annotations.noseTip; // point        
        const noseLeftCorner = annotations.noseLeftCorner; // point
        const noseRightCorner = annotations.noseRightCorner; // point
        
        const midwayBetweenEyes = annotations.midwayBetweenEyes; // point
        const leftCheek = annotations.leftCheek; // point
        const rightCheek  = annotations.rightCheek; // point

        drawFeaturePoint(noseBottom, 6);
        drawFeaturePoint(noseTip, 6);
        drawFeaturePoint(noseLeftCorner, 6);
        drawFeaturePoint(noseRightCorner, 6);
        drawFeaturePoint(midwayBetweenEyes, 6);
        drawFeaturePoint(leftCheek, 6);
        drawFeaturePoint(rightCheek, 6);
        
    }
}

function drawFeatureLine(feature, closed) {
    for (let i = 0; i < feature.length; i += 1) {
        noFill();
        stroke(255,255,255);
        if (i<(feature.length-1)) {
            const [x1, y1] = feature[i];
            const [x2, y2] = feature[i+1];
            line(x1*aspectX,y1*aspectY,x2*aspectX,y2*aspectY);
        } else {
            const [x1, y1] = feature[i];
            const [x2, y2] = feature[0];
            if (closed == true) line(x1*aspectX,y1*aspectY,x2*aspectX,y2*aspectY);
        }
    }
}

function drawFeaturePoint(feature, size) {
    for (let i = 0; i < feature.length; i += 1) {
        const [x, y] = feature[i]
        noStroke();
        fill(255, 255, 255);
        ellipse(x*aspectX, y*aspectY, size, size);
    }
}

/* draw ellipses over the detected FaceMesh keypoints */
function drawFaceMeshKeypoints() {
    // var mv = document.getElementById("v");
    // aspectX = windowWidth / mv.width;
    // aspectY = windowHeight / mv.height;
    closed = true;
    for (let i = 0; i < predictions.length; i += 1) {
      // console.log(predictions);
      const keypoints = predictions[i].scaledMesh;

      /* draw facial mesh lines */
    //   noFill();
    //   stroke(255,255,255);
    //   for (let j = 0; j < keypoints.length; j += 1) {

    //     if (j<(keypoints.length-1)) {
    //         const [x1, y1] = keypoints[j];
    //         const [x2, y2] = keypoints[j+1];
    //         line(x1*aspectX,y1*aspectY,x2*aspectX,y2*aspectY);
    //     } else {
    //         const [x1, y1] = keypoints[j];
    //         const [x2, y2] = keypoints[0];
    //         if (closed == true) line(x1*aspectX,y1*aspectY,x2*aspectX,y2*aspectY);
    //     }
    //   }
  
      /* draw facial keypoints */
      for (let j = 0; j < keypoints.length; j += 1) {
        const [x, y] = keypoints[j];
  
        noStroke();
        fill(255, 255, 255);
        ellipse(x*aspectX, y*aspectY, 3, 3);
      }
    }
}

/* draw FaceMesh bounding box */
function drawFaceMeshBoundingBox() {
    // var mv = document.getElementById("v");
    // aspectX = windowWidth / mv.width;
    // aspectY = windowHeight / mv.height;
    for (let i = 0; i < predictions.length; i += 1) {
      const boundingBox = predictions[i].boundingBox;
      const [x1, y1] = boundingBox.topLeft[0];
      const [x2, y2] = boundingBox.bottomRight[0];

      /* draw bounding box */
      stroke(255, 255, 255);
      noFill();
      rect(x1*aspectX, y1*aspectY, (x2-x1)*aspectX, (y2-y1)*aspectY);
      // console.log((x1+x2)/2+"-"+(y1+y2)/2);
    }
}

/* draw CLM points */
function drawCLMPoints() {
    for (var i=0; i<positions.length -1; i++) {
        // set the color of the ellipse based on position on screen
        fill(map(positions[i][0], width*0.33, width*0.66, 0, 255), map(positions[i][1], height*0.33, height*0.66, 0, 100), 0);
        
        // draw ellipse
        noStroke();
        ellipse(positions[i][0], positions[i][1], 10, 10);
        
        // draw line
        stroke(map(positions[i][0], width*0.33, width*0.66, 0, 255), map(positions[i][1], height*0.33, height*0.66, 0, 100), 0,50);
        // stroke(255);
        line(positions[i][0], positions[i][1], positions[i+1][0], positions[i+1][1]);
    }
}

function keyPressed() {
    if (keyCode === 80) { // P
        showCLMPoints = !showCLMPoints;
    } else if (keyCode === 73) { // I
        showImage = !showImage;
    } else if (keyCode === 77) { // M
        showFaceMesh = !showFaceMesh;
    } else if (keyCode === 66) { // B
        showBoundingBox = !showBoundingBox;
    } else if (keyCode === 83) { // S
        // showFullscreen = !showFullscreen;
        fullscreen(!showFullscreen);
    } else if (keyCode === 84) { // T
        showText = !showText;
    } else if (keyCode == 70) { // F
        showFeatures = !showFeatures;
    } else if (keyCode === 76) { // L
        console.log(uuidv4());
    }
    return false; // prevent any default behavior
  }

  function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
  }
