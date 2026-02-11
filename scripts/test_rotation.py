from scipy.spatial.transform import Rotation as R

# Use uppercase 'XYZ' for Intrinsic rotations (matches Isaac Sim / USD)
# r = R.from_euler('xyz', [-34, -26, -140], degrees=True)
r = R.from_euler('XYZ', [180, -28, -90], degrees=True)

quat_xyzw = r.as_quat() 
quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

# Using a list comprehension to round, then printing the list directly
result = [round(float(x), 3) for x in quat_wxyz]
print(result)


# Convention,transforms3d Name,Scipy Name,Math Description,Used In
# Static / Extrinsic,sxyz,'xyz',R=Rz​⋅Ry​⋅Rx​,"Camera setups, World transforms"
# Rotating / Intrinsic,rxyz,'XYZ',R=Rx​⋅Ry​⋅Rz​,"Isaac Sim, ROS URDFs, Aerospace"