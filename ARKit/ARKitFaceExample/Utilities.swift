/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Convenience extensions for system types.
*/

import ARKit
import SceneKit
import simd

extension SCNMatrix4 {
    /**
     Create a 4x4 matrix from CGAffineTransform, which represents a 3x3 matrix
     but stores only the 6 elements needed for 2D affine transformations.
     
     [ a  b  0 ]     [ a  b  0  0 ]
     [ c  d  0 ]  -> [ c  d  0  0 ]
     [ tx ty 1 ]     [ 0  0  1  0 ]
     .               [ tx ty 0  1 ]
     
     Used for transforming texture coordinates in the shader modifier.
     (Needs to be SCNMatrix4, not SIMD float4x4, for passing to shader modifier via KVC.)
     */
    init(_ affineTransform: CGAffineTransform) {
        self.init()
        m11 = Float(affineTransform.a)
        m12 = Float(affineTransform.b)
        m21 = Float(affineTransform.c)
        m22 = Float(affineTransform.d)
        m41 = Float(affineTransform.tx)
        m42 = Float(affineTransform.ty)
        m33 = 1
        m44 = 1
    }
}

extension SCNReferenceNode {
    convenience init(named resourceName: String, loadImmediately: Bool = true) {
        let url = Bundle.main.url(forResource: resourceName, withExtension: "scn", subdirectory: "Models.scnassets")!
        self.init(url: url)!
        if loadImmediately {
            self.load()
        }
    }
}

/* Credits to https://michael-martinez.fr/arkit-transform-matrices-quaternions-and-related-conversions/ */

public extension matrix_float4x4 {
    
    // Retrieve position
    
    public var position: SCNVector3 {
        get {
            return SCNVector3Make(columns.3.x, columns.3.y, columns.3.z)
        }
    }
    
    // Get Euler Angles
    
    public var eulerAngles: SCNVector3 {
        get {
            // get quaternions
            // robust Alternative method, uses sign() function from simd library
            // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            let half: Float = 0.5
            let A = matrix_float4x4(rows: [
                                    simd_float4(half, half, half, half),
                                    simd_float4(half, half, -half, -half),
                                    simd_float4(half, -half, half, -half),
                                    simd_float4(half, -half, -half, half)])
            let x = simd_float4(1.0, columns.0.x, columns.1.y, columns.2.z)
            // initial quaternion multiplication
            var Q : simd_float4 = _simd_pow_f4(max(x * A, 0.0), simd_float4(half, half, half, half)) / 2.0
            let getSigns = simd_sign(simd_float4(1,
                                            columns.2.y - columns.1.z,
                                            columns.0.z - columns.2.x,
                                            columns.1.x - columns.0.y))
            Q = getSigns * Q
            
            let sinr = 2.0 * (Q.w * Q.x + Q.y * Q.z)
            let cosr = 1.0 - 2.0 * (Q.x * Q.x + Q.y * Q.y)
            let sinp = 2.0 * (Q.w * Q.y - Q.z * Q.x)
            let siny = 2.0 * (Q.w * Q.z + Q.x * Q.y)
            let cosy = 1.0 - 2.0 * (Q.y * Q.y + Q.z * Q.z)
            
            // get roll (x-axis rot)
            let roll = atan2(sinr, cosr)
            
            // get pitch (y-axis rot)
            var pitch: Float
            if abs(sinp) >= 1 {
                pitch = copysign(Float.pi / 2.0, sinp)
            } else {
                pitch = asin(sinp)
            }
            
            // yaw (z-axis rot)
            let yaw = atan2(siny, cosy)
            
            return SCNVector3(pitch, yaw, roll)
        }
    }
}
