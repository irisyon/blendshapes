//
//  UserData.swift
//  ARKitFaceExample
//
//  Created by Iris Yon on 11/3/18.
//  Copyright © 2018 Apple. All rights reserved.
//

import Foundation
import SceneKit
import ARKit

class UserData: NSObject, VirtualContentController {
    
    var contentNode: SCNNode?
    var Output = ""
    var presetKeys: [String] = []
    var BlendShapeKeyOrdering: [ARFaceAnchor.BlendShapeLocation] = []
    
    // Load multiple copies of the axis origin visualization for the transforms this class visualizes.
    lazy var rightEyeNode = SCNReferenceNode(named: "coordinateOrigin")
    lazy var leftEyeNode = SCNReferenceNode(named: "coordinateOrigin")
    
    // Load properties for blendshape preview
    private var originalJawY: Float = 0
    
    private lazy var jawNode = contentNode!.childNode(withName: "jaw", recursively: true)!
    private lazy var eyeLeftNode = contentNode!.childNode(withName: "eyeLeft", recursively: true)!
    private lazy var eyeRightNode = contentNode!.childNode(withName: "eyeRight", recursively: true)!
    
    private lazy var jawHeight: Float = {
        let (min, max) = jawNode.boundingBox
        return max.y - min.y
    }()
    
    /// - Tag: ARTracking
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let sceneView = renderer as? ARSCNView,
            let faceAnchor = anchor as? ARFaceAnchor,
            anchor is ARFaceAnchor else { return nil }
        
        let faceGeometry = ARSCNFaceGeometry(device: sceneView.device!)!
        let material = faceGeometry.firstMaterial!
        
        material.diffuse.contents = #imageLiteral(resourceName: "wireframeTexture") // Example texture map image.
        material.lightingModel = .physicallyBased
        
        contentNode = SCNNode(geometry: faceGeometry)
        // Add content for eye tracking in iOS 12.
        self.addEyeTransformNodes()
        // Create labels for CSV output
        //        Output = "face_position_x,face_position_y,face_position_z,"
        //            + "face_orientation_x,face_orientation_y,face_orientation_z,"
        //            + "L_eye_orientation_x,L_eye_orientation_y,L_eye_orientation_z,"
        //            + "R_eye_orientation_x,R_eye_orientation_y,R_eye_orientation_z"
        //        presetKeys = Output.components(separatedBy: ",")
        for (key, _) in faceAnchor.blendShapes {
            Output += "," + key.rawValue
            BlendShapeKeyOrdering.append(key)
        }
        Output.remove(at: Output.startIndex)
        Output += "\n"
        return contentNode
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor,
            let faceGeometry = node.geometry as? ARSCNFaceGeometry
            else { return }
        
        rightEyeNode.simdTransform = faceAnchor.rightEyeTransform
        leftEyeNode.simdTransform = faceAnchor.leftEyeTransform
        faceGeometry.update(from: faceAnchor.geometry)
        //        let blendShapes = faceAnchor.blendShapes
        //        guard let eyeBlinkLeft = blendShapes[.eyeBlinkLeft] as? Float,
        //            let eyeBlinkRight = blendShapes[.eyeBlinkRight] as? Float,
        //            let jawOpen = blendShapes[.jawOpen] as? Float
        //            else { return }
        //        eyeLeftNode.scale.z = 1 - eyeBlinkLeft
        //        eyeRightNode.scale.z = 1 - eyeBlinkRight
        //        jawNode.position.y = originalJawY - jawHeight * jawOpen
        // TODO: Add in weights manually for face & eye transformation, orientation
        var tempOut = ""
        for key in BlendShapeKeyOrdering {
            tempOut += "," + (faceAnchor.blendShapes[key]?.stringValue)!
        }
        tempOut.remove(at: tempOut.startIndex)
        Output += tempOut + "\n"
    }
    func addEyeTransformNodes() {
        guard #available(iOS 12.0, *), let anchorNode = contentNode else { return }
        
        // Scale down the coordinate axis visualizations for eyes.
        rightEyeNode.simdPivot = float4x4(diagonal: float4(3, 3, 3, 1))
        leftEyeNode.simdPivot = float4x4(diagonal: float4(3, 3, 3, 1))
        
        anchorNode.addChildNode(rightEyeNode)
        anchorNode.addChildNode(leftEyeNode)
    }

    
    
  
    
    func exportData() {
        
        let fileName = "data.csv"
        //let path = NSURL(fileURLWithPath: .documentDirectory).appendingPathComponent(fileName)
        
        var csvText = Output
        
        do {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            let documentsDirectory = paths[0]
            //let fileManager = FileManager.default
            /*
            let path = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            let fileURL = path.appendingPathComponent(fileName)
           // let documentDirectoryURL =  try! FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)

            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)*/
            //try csvText.write(to: path!, atomically: true, encoding: String.Encoding.utf8)
          // print(fileURL)
          /*  var path = try! FileManager.default.url(for: .documentDirectory, in: .UserDomainMask, appropriateFor: nil, create: false)
                .appendingPathComponent("cardsFile.csv")

            print(path)
*/
            let documentsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
            let fileURL = documentsURL.appendingPathComponent(fileName)
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            //print(documentsDirectory)
        } catch {
            print("Failed to create file")
            print("\(error)")
        }
        //print(path ?? "not found")
        
    }
   
  
}


