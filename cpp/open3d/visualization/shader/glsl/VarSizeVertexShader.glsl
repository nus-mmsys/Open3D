#version 330

in vec3 vertex_position;
in vec3 vertex_color;
in float vertex_size;
uniform mat4 MVP;

out vec3 fragment_color;

void main()
{
    gl_Position = MVP * vec4(vertex_position, 1);
	gl_PointSize = vertex_size;
    fragment_color = vertex_color;
}
