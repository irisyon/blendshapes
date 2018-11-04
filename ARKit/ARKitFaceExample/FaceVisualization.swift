/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Displays coordinate axes visualizing the tracked face pose (and eyes in iOS 12).
*/

import ARKit
import SceneKit

class FaceVisualization: NSObject {
    
    var contentNode: SCNNode?
    var Output = ""
    
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
        
//        #if targetEnvironment(simulator)
//        #error("ARKit is not supported in iOS Simulator. Connect a physical iOS device and select it as your Xcode run destination, or select Generic iOS Device as a build-only destination.")
//        #else
        let faceGeometry = ARSCNFaceGeometry(device: sceneView.device!)!
        let material = faceGeometry.firstMaterial!
        
        material.diffuse.contents = #imageLiteral(resourceName: "wireframeTexture") // Example texture map image.
        material.lightingModel = .physicallyBased
        
        contentNode = SCNNode(geometry: faceGeometry)
        // Add content for eye tracking in iOS 12.
        self.addEyeTransformNodes()
        // Create labels for CSV output
        Output = "face_position, face_orientation, L_eye_orientation, R_eye_orientation"
        for (key, _) in faceAnchor.blendShapes {
            Output += ", " + key.rawValue;
        }
        Output += "\n"
//        #endif
        return contentNode
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard #available(iOS 12.0, *), let faceAnchor = anchor as? ARFaceAnchor
            else { return }
        
        rightEyeNode.simdTransform = faceAnchor.rightEyeTransform
        leftEyeNode.simdTransform = faceAnchor.leftEyeTransform
        let blendShapes = faceAnchor.blendShapes
        guard let eyeBlinkLeft = blendShapes[.eyeBlinkLeft] as? Float,
            let eyeBlinkRight = blendShapes[.eyeBlinkRight] as? Float,
            let jawOpen = blendShapes[.jawOpen] as? Float
            else { return }
        eyeLeftNode.scale.z = 1 - eyeBlinkLeft
        eyeRightNode.scale.z = 1 - eyeBlinkRight
        jawNode.position.y = originalJawY - jawHeight * jawOpen
        // TODO: Add in weights manually for face & eye transformation, orientation
        for (_, weight) in faceAnchor.blendShapes {
            Output += ", " + weight.stringValue
        }
        Output += "\n"
    }
    
    func addEyeTransformNodes() {
        guard #available(iOS 12.0, *), let anchorNode = contentNode else { return }
        
        // Scale down the coordinate axis visualizations for eyes.
        rightEyeNode.simdPivot = float4x4(diagonal: float4(3, 3, 3, 1))
        leftEyeNode.simdPivot = float4x4(diagonal: float4(3, 3, 3, 1))
        
        anchorNode.addChildNode(rightEyeNode)
        anchorNode.addChildNode(leftEyeNode)
    }
    
    func exportCSV() {
        
    }

}
