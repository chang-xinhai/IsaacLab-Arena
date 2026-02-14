import os
import xml.etree.ElementTree as ET

# --- 配置路径 ---
# 根据你提供的信息设置的路径
BASE_DIR = "/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_assets/object/microwave_7221"
URDF_PATH = os.path.join(BASE_DIR, "mobility.urdf")
MESH_DIR = os.path.join(BASE_DIR, "textured_objs")

def sanitize(name):
    """将连字符替换为下划线，修复非法字符"""
    if name:
        return name.replace("-", "_")
    return name

def fix_files_in_directory():
    print(f"--- 1. Processing files in {MESH_DIR} ---")
    
    # 获取目录下所有文件
    all_files = os.listdir(MESH_DIR)
    
    # 步骤 A: 重命名所有文件 (包括 .obj 和 .mtl)
    renamed_map = {} # 记录 old_name -> new_name
    
    for filename in all_files:
        if "-" in filename:
            old_path = os.path.join(MESH_DIR, filename)
            new_filename = sanitize(filename)
            new_path = os.path.join(MESH_DIR, new_filename)
            
            # 重命名文件
            os.rename(old_path, new_path)
            renamed_map[filename] = new_filename
            print(f"Renamed: {filename} -> {new_filename}")
        else:
            renamed_map[filename] = filename

    # 步骤 B: 修复 .obj 文件内部的 .mtl 引用
    # 因为我们重命名了 .mtl 文件，.obj 内部的 "mtllib name.mtl" 也需要更新
    print("\n--- 2. Updating .mtl references inside .obj files ---")
    
    # 重新获取文件列表（因为刚才重命名了）
    current_files = os.listdir(MESH_DIR)
    
    for filename in current_files:
        if filename.endswith(".obj"):
            file_path = os.path.join(MESH_DIR, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                modified = False
                new_lines = []
                
                for line in lines:
                    if line.strip().startswith("mtllib"):
                        # 解析: mtllib original-1.mtl
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            mtl_name = parts[1]
                            if "-" in mtl_name:
                                new_mtl_name = sanitize(mtl_name)
                                line = f"mtllib {new_mtl_name}\n"
                                modified = True
                    new_lines.append(line)
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    print(f"Fixed mtllib in: {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def fix_urdf():
    print(f"\n--- 3. Processing URDF: {URDF_PATH} ---")
    
    try:
        tree = ET.parse(URDF_PATH)
        root = tree.getroot()
        
        # A. 修复所有标签的 'name' 属性 (Visual, Collision, Joint, Link)
        for elem in root.iter():
            if 'name' in elem.attrib:
                original_name = elem.attrib['name']
                if "-" in original_name:
                    new_name = sanitize(original_name)
                    elem.attrib['name'] = new_name
                    print(f"URDF Name: '{original_name}' -> '{new_name}'")

        # B. 修复 Mesh 的 'filename' 路径
        for mesh in root.iter('mesh'):
            if 'filename' in mesh.attrib:
                full_path = mesh.attrib['filename']
                head, tail = os.path.split(full_path)
                
                # 如果文件名部分包含 "-"，则替换它
                if "-" in tail:
                    new_tail = sanitize(tail)
                    # 重新组合路径
                    new_full_path = os.path.join(head, new_tail)
                    mesh.attrib['filename'] = new_full_path
                    print(f"URDF Mesh Path: .../{tail} -> .../{new_tail}")

        # 保存为新文件
        new_urdf_path = URDF_PATH.replace(".urdf", "_fixed.urdf")
        tree.write(new_urdf_path, encoding='utf-8', xml_declaration=True)
        print(f"\nSUCCESS! Saved fixed URDF to: {new_urdf_path}")
        print("Please import this new file in Isaac Sim.")
        
    except Exception as e:
        print(f"Error parsing URDF: {e}")

if __name__ == "__main__":
    if not os.path.exists(MESH_DIR):
        print(f"Error: Mesh directory not found: {MESH_DIR}")
    elif not os.path.exists(URDF_PATH):
        print(f"Error: URDF file not found: {URDF_PATH}")
    else:
        # 1. 先修复文件系统中的文件名
        fix_files_in_directory()
        # 2. 再修复 URDF 中的引用
        fix_urdf()