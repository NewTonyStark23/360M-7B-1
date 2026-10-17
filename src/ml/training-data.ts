// ============================================================
// SMOL-LM Training Data
// Built-in corpus for quick training in-browser
// ============================================================

export const TRAINING_CORPORA: Record<string, { name: string; text: string; description: string }> = {
  simple_english: {
    name: 'Simple English',
    description: 'Basic English sentences and patterns',
    text: `the cat sat on the mat. the dog ran in the park. the bird flew over the tree. the fish swam in the sea. the sun is bright today. the moon shines at night. the stars twinkle in the sky. the rain falls on the ground. the wind blows through the trees. the snow covers the mountain. the river flows to the ocean. the lake is calm and still. the boy went to school. the girl read a book. the man walked down the street. the woman drove her car. the child played with toys. the baby slept in the crib. the teacher taught the class. the student learned new things. the doctor helped the patient. the farmer grew the crops. the baker made fresh bread. the cook prepared the meal. the singer performed a song. the dancer moved gracefully. the painter created art. the writer told a story. the builder constructed the house. the gardener planted flowers. the baker baked a cake. the farmer fed the animals. the teacher graded papers. the student studied hard. the doctor examined the patient. the nurse gave medicine. the pilot flew the plane. the driver stopped at the light. the police officer directed traffic. the firefighter put out the fire. the soldier protected the country. the sailor navigated the sea. the astronaut explored space. the scientist conducted experiments. the engineer designed the bridge. the artist painted a picture. the musician played the guitar. the actor performed on stage. the director made a movie. the photographer took pictures. the journalist wrote an article. the editor reviewed the text. the publisher printed the book. the librarian organized the shelves. the cashier counted the money. the manager supervised the team. the owner ran the business. the customer bought the product. the client hired the lawyer. the patient visited the clinic. the guest checked into the hotel. the tourist explored the city. the visitor admired the museum. the fan cheered for the team. the player scored a goal. the coach trained the athletes. the referee made the call. the umpire watched the game. the crowd enjoyed the show. the audience applauded loudly. the host welcomed the guests. the speaker gave a speech. the judge heard the case. the jury reached a verdict. the lawyer presented evidence. the witness told the truth. the detective solved the mystery. the criminal committed the crime. the hero saved the day. the villain caused trouble. the friend offered support. the neighbor borrowed sugar. the relative visited for dinner. the couple went on vacation. the family gathered for holidays. the community celebrated together. the nation elected a leader. the world changed forever. time moves forward always. life is full of surprises. love conquers all obstacles. knowledge is power indeed. wisdom comes with experience. patience brings great rewards. courage overcomes all fear. hope shines in darkness. faith moves mountains high. truth sets the spirit free. peace begins with a smile. joy fills the grateful heart. beauty surrounds us always. nature heals the weary soul. music soothes the troubled mind. art expresses the inner world. words have the power to change. ideas shape the world we live in. dreams become reality through action. every day is a new beginning. the journey matters more than the destination. what we give returns to us. the best things in life are free. together we are stronger than alone. knowledge shared is knowledge doubled. the future belongs to the curious. small steps lead to great changes. the pen is mightier than the sword. actions speak louder than words. practice makes perfect over time. every expert was once a beginner. the early bird catches the worm. fortune favors the bold and brave. where there is a will there is a way. all good things come to those who wait. the proof of the pudding is in eating. a picture is worth a thousand words. necessity is the mother of invention. two heads are better than one. the grass is always greener elsewhere. better late than never they say. honesty is the best policy always. silence is golden in many cases. the early worm gets the bird too. time and tide wait for no one. a stitch in time saves nine later. look before you leap they advise. where there is smoke there is fire. every cloud has a silver lining. the best is yet to come for us. what goes around comes around again. you reap what you sow in life. the truth will set you free at last. knowledge is the key to success. imagination is more important than knowledge. the only limit is your own mind. believe in yourself and you can do it. every moment is a fresh beginning. life is what you make of it. the greatest adventure is the one ahead. today is the first day of the rest. be the change you want to see. in the end we only regret chances not taken. the purpose of life is to find your gift. the meaning of life is to give it meaning.`,
  },
  code_patterns: {
    name: 'Code Patterns',
    description: 'Simple programming patterns and syntax',
    text: `function add(a, b) { return a + b; }
function sub(a, b) { return a - b; }
function mul(a, b) { return a * b; }
function div(a, b) { return a / b; }
const x = 10;
const y = 20;
const z = x + y;
let count = 0;
count = count + 1;
for (let i = 0; i < 10; i++) { console.log(i); }
if (x > 5) { console.log("big"); } else { console.log("small"); }
while (count < 100) { count++; }
const arr = [1, 2, 3, 4, 5];
arr.push(6);
arr.pop();
const obj = { name: "test", value: 42 };
console.log(obj.name);
function greet(name) { return "hello " + name; }
const result = greet("world");
class Animal { constructor(name) { this.name = name; } speak() { return this.name + " makes a sound"; } }
class Dog extends Animal { speak() { return this.name + " barks"; } }
const dog = new Dog("Rex");
console.log(dog.speak());
const square = (x) => x * x;
const cube = (x) => x * x * x;
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
const sum = numbers.reduce((a, b) => a + b, 0);
async function fetchData(url) { const response = await fetch(url); const data = await response.json(); return data; }
try { const result = riskyOperation(); } catch (error) { console.error(error); }
const promise = new Promise((resolve, reject) => { resolve("done"); });
export default function main() { console.log("hello world"); }
import { useState } from "react";
function Counter() { const [count, setCount] = useState(0); return <button onClick={() => setCount(count + 1)}>{count}</button>; }
const div_style = { color: "red", fontSize: "16px" };
const template = \`hello \${name}, you are \${age} years old\`;
if (typeof x === "string") { console.log("string"); }
const { name, age } = person;
const [first, ...rest] = array;
const merged = { ...obj1, ...obj2 };
const condition ? value1 : value2;
null ?? "default";
0 || "fallback";
"" && "won't show";
typeof undefined === "undefined";
Array.isArray([1, 2, 3]);
Object.keys({ a: 1, b: 2 });
JSON.stringify({ key: "value" });
parseInt("42"); parseFloat("3.14");
Math.max(1, 2, 3); Math.min(1, 2, 3);
Math.random(); Math.floor(3.7); Math.ceil(3.2);
"hello".toUpperCase(); "WORLD".toLowerCase();
"hello world".split(" ");
"  trimmed  ".trim();
"hello".includes("ell");
"hello".startsWith("he");
"hello".endsWith("lo");
"hello".replace("l", "r");
Date.now(); new Date().toISOString();
setTimeout(() => {}, 1000);
setInterval(() => {}, 1000);
clearTimeout(timerId); clearInterval(intervalId);
localStorage.setItem("key", "value");
localStorage.getItem("key");
document.getElementById("app");
document.querySelector(".class");
element.addEventListener("click", handler);
element.classList.add("active");
element.style.display = "none";
window.innerWidth; window.innerHeight;
navigator.userAgent;
console.log("debug"); console.warn("warning"); console.error("error");`,
  },
  poetry: {
    name: 'Simple Poetry',
    description: 'Simple rhyming verses and poetic patterns',
    text: `roses are red and violets are blue. the sky is above and the ocean is too.
the sun shines bright in the morning light. the stars come out when day turns to night.
a bird in the hand is worth two in the sky. the days go by as the clouds float high.
the wind whispers low through the trees that grow. the river will flow wherever it goes.
in the morning dew the world feels new. under skies of blue the dreams come true.
the moon glows white in the silent night. everything feels so calm and right.
spring brings the flowers after winter hours. summer has its powers with sunny showers.
autumn leaves fall like nature's call. winter stands tall with its icy wall.
the cat sat still on the window sill. time stood still with a gentle thrill.
the dog ran fast through the green grass. watching the clouds as they slowly pass.
the ocean waves crash upon the shore. always returning for so much more.
the mountain stands proud above the cloud. its peak so high and its voice so loud.
the forest is deep with secrets to keep. while the world above is fast asleep.
the garden blooms with colorful perfumes. chasing away the winter glooms.
the candle flickers with golden light. pushing back the edges of the night.
the rain falls soft from the clouds aloft. lifting our spirits way up aloft.
the snow falls white covering all in sight. a winter wonderland pure and bright.
the fire burns warm through the winter storm. keeping us safe from the weather's norm.
the book tells tales of winds and gales. of ships with billowing sails.
the pen writes words that fly like birds. the sweetest songs of feathered herds.
the music plays a gentle tune. under the light of the harvest moon.
the dance of leaves in autumn breeze. the whisper of the ancient trees.
the laughter of a child at play. the warmth of a sunny day.
the kindness of a stranger's smile. makes the journey all the while.
the love between a mother and child. a bond that makes the heart run wild.
the friendship built on trust and care. a treasure beyond all compare.
the courage found in darkest hour. the inner strength the inner power.
the wisdom learned from years gone by. the truth that never tells a lie.
the beauty found in simple things. the joy that every morning brings.
the peace that comes at end of day. when all the world is far away.
the hope that lives within the heart. a brand new place to make a start.`,
  },
};

export function getTrainingChunks(text: string, chunkSize: number = 64, stride: number = 32): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length - chunkSize; i += stride) {
    chunks.push(text.slice(i, i + chunkSize));
  }
  return chunks;
}
